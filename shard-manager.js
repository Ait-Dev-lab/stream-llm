// browser/shard-manager.js
// StreamWeights — the core orchestration layer
// Handles fetch → upload → compute → evict pipeline

class StreamWeightManager {
    constructor(config) {
        this.jumpstartUrl = config.jumpstartUrl;
        this.device = null;
        this.activeBuffers = new Map();
        this.shaderModule = null;
        this.modelConfig = null;
        this.computePipeline = null;
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported. Please use Chrome 113+ or Edge 113+.");
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU adapter found");
        
        // Check device limits before requesting
        const requiredSize = 512 * 1024 * 1024;
        const supportedSize = adapter.limits.maxStorageBufferBindingSize;
        
        if (supportedSize < requiredSize) {
            throw new Error(`GPU insufficient: needs ${requiredSize / 1024 / 1024}MB, has ${supportedSize / 1024 / 1024}MB`);
        }
        
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: requiredSize,
                maxBufferSize: requiredSize
            }
        });
        
        // Create shader module for inference
        await this.initShaders();
        
        console.log("StreamWeightManager: WebGPU initialized");
    }

    async initShaders() {
        const shaderCode = `
            @group(0) @binding(0) var<storage, read> inputWeights: array<f32>;
            @group(0) @binding(1) var<storage, read> inputTokens: array<u32>;
            @group(0) @binding(2) var<storage, read_write> outputLogits: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if (idx >= arrayLength(&outputLogits)) { return; }
                
                // Simplified inference: weight * token embedding
                var sum = 0.0;
                for (var i = 0u; i < 1024u; i++) {
                    if (idx * 1024u + i < arrayLength(&inputWeights)) {
                        sum += inputWeights[idx * 1024u + i];
                    }
                }
                outputLogits[idx] = sum / 1024.0;
            }
        `;
        
        const shaderModule = this.device.createShaderModule({ code: shaderCode });
        this.computePipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: { module: shaderModule, entryPoint: "main" }
        });
        
        console.log("StreamWeightManager: Shaders initialized");
    }

    async jumpstart(prompt) {
        console.log("Jumpstart: Requesting initial tokens...");
        const response = await fetch(`${this.jumpstartUrl}/jumpstart`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, maxInitialTokens: 8 })
        });
        
        if (!response.ok) throw new Error(`Jumpstart failed: ${response.status}`);
        
        const data = await response.json();
        this.modelConfig = data.modelConfig;
        
        console.log(`Jumpstart: ${data.initialTokens.length} tokens received, ${data.modelConfig.shardCount} shards available`);
        return data;
    }

    async fetchShard(shardId) {
        const shard = this.modelConfig.shards.find(s => s.id === shardId);
        if (!shard) throw new Error(`Shard ${shardId} not found`);
        
        console.log(`Fetching shard ${shardId} from ${shard.url}`);
        const response = await fetch(shard.url);
        if (!response.ok) throw new Error(`Shard ${shardId} fetch failed`);
        
        return await response.arrayBuffer();
    }

    async processLayerWithGPU(weightsData, inputTokens) {
        // Upload weights to GPU
        const weightsBuffer = this.device.createBuffer({
            size: weightsData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        this.device.queue.writeBuffer(weightsBuffer, 0, weightsData);
        
        // Upload input tokens
        const tokenArray = new Uint32Array(inputTokens);
        const tokenBuffer = this.device.createBuffer({
            size: tokenArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        
        this.device.queue.writeBuffer(tokenBuffer, 0, tokenArray);
        
        // Output buffer for logits
        const outputSize = 32000 * 4; // Vocabulary size * 4 bytes (f32)
        const outputBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        
        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: weightsBuffer } },
                { binding: 1, resource: { buffer: tokenBuffer } },
                { binding: 2, resource: { buffer: outputBuffer } }
            ]
        });
        
        // Execute compute shader
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(outputSize / 256 / 4));
        passEncoder.end();
        
        const commandBuffer = commandEncoder.finish();
        this.device.queue.submit([commandBuffer]);
        
        // Wait for GPU to finish
        await this.device.queue.onSubmittedWorkDone();
        
        // Read back results
        const readBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputSize);
        this.device.queue.submit([copyEncoder.finish()]);
        
        await readBuffer.mapAsync(GPUMapMode.READ);
        const logits = new Float32Array(readBuffer.getMappedRange());
        
        // Find the token with highest probability (argmax)
        let maxIdx = 0;
        for (let i = 1; i < logits.length; i++) {
            if (logits[i] > logits[maxIdx]) maxIdx = i;
        }
        
        readBuffer.unmap();
        
        // Cleanup
        weightsBuffer.destroy();
        tokenBuffer.destroy();
        outputBuffer.destroy();
        readBuffer.destroy();
        
        return maxIdx; // Return the predicted token ID
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async runInference(prompt, onToken) {
        const jumpstartData = await this.jumpstart(prompt);
        
        // Stream the first 8 tokens from jumpstart
        for (let i = 0; i < jumpstartData.initialTokens.length; i++) {
            onToken(jumpstartData.initialTokens[i], i === 0);
            await this.delay(50); // Natural streaming speed
        }
        
        console.log(`Inference: Browser takes over at layer ${jumpstartData.nextLayer}`);
        
        let currentTokens = jumpstartData.tokenIds || [];
        let generatedTokens = jumpstartData.initialTokens.length;
        const maxTokens = 200; // Limit to avoid infinite generation
        
        // Process remaining layers and generate tokens
        for (let layer = jumpstartData.nextLayer; layer < this.modelConfig.totalLayers; layer++) {
            console.log(`Processing layer ${layer + 1}/${this.modelConfig.totalLayers}`);
            
            const shardData = await this.fetchShard(`layer_${layer}`);
            const layerData = new Float32Array(shardData);
            
            // For demo purposes, generate a token for each layer
            // In a real implementation, this would process the entire sequence
            if (generatedTokens < maxTokens) {
                const nextTokenId = await this.processLayerWithGPU(layerData, currentTokens);
                
                // Convert token ID to text (simplified - you'll need a real tokenizer)
                const nextToken = this.decodeToken(nextTokenId);
                
                // Stream the generated token
                onToken(nextToken, false);
                generatedTokens++;
                currentTokens.push(nextTokenId);
                
                await this.delay(30); // Smooth streaming
            }
        }
        
        onToken("\n\n[Streaming complete]", false);
        console.log("Inference complete");
    }
    
    decodeToken(tokenId) {
        // This is a simplified placeholder
        // In production, use the actual tokenizer from your model config
        const fakeTokens = ["the", "a", "an", "model", "language", "neural", "network", "deep", "learning", "AI"];
        if (tokenId < fakeTokens.length) {
            return fakeTokens[tokenId];
        }
        return `[${tokenId}]`;
    }
    
    // Helper to clean up resources
    destroy() {
        for (const buffer of this.activeBuffers.values()) {
            buffer.destroy();
        }
        this.activeBuffers.clear();
    }
}

export { StreamWeightManager };