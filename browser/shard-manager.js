// browser/shard-manager.js
// StreamWeights — the core orchestration layer

class StreamWeightManager {
    constructor(config) {
        this.jumpstartUrl = config.jumpstartUrl;
        this.device = null;
        this.activeBuffers = new Map();
        this.shaderModule = null;
        this.modelConfig = null;
        this.computePipeline = null;
        
        // CDN configuration
        this.cdnBase = "https://github.com/Ait-Dev-lab/stream-llm/releases/download/v1.0.0";
        this.totalShards = 30; // 0 to 29
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported. Please use Chrome 113+ or Edge 113+.");
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU adapter found");
        
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
        
        console.log(`Jumpstart: ${data.initialTokens.length} tokens received`);
        return data;
    }

    async fetchShard(layerIndex) {
        const shardName = `shard_${String(layerIndex).padStart(3, '0')}.bin`;
        const shardUrl = `${this.cdnBase}/${shardName}`;
        
        console.log(`Fetching shard ${layerIndex}: ${shardName}`);
        const response = await fetch(shardUrl);
        if (!response.ok) throw new Error(`Shard ${layerIndex} failed: ${response.status}`);
        
        return await response.arrayBuffer();
    }

    async processLayerWithGPU(weightsData, inputTokens) {
        const weightsBuffer = this.device.createBuffer({
            size: weightsData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(weightsBuffer, 0, weightsData);
        
        const tokenArray = new Uint32Array(inputTokens);
        const tokenBuffer = this.device.createBuffer({
            size: tokenArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(tokenBuffer, 0, tokenArray);
        
        const outputSize = 32000 * 4;
        const outputBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        
        const bindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: weightsBuffer } },
                { binding: 1, resource: { buffer: tokenBuffer } },
                { binding: 2, resource: { buffer: outputBuffer } }
            ]
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(outputSize / 256 / 4));
        passEncoder.end();
        
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        
        const readBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputSize);
        this.device.queue.submit([copyEncoder.finish()]);
        
        await readBuffer.mapAsync(GPUMapMode.READ);
        const logits = new Float32Array(readBuffer.getMappedRange());
        
        let maxIdx = 0;
        for (let i = 1; i < logits.length; i++) {
            if (logits[i] > logits[maxIdx]) maxIdx = i;
        }
        
        readBuffer.unmap();
        
        weightsBuffer.destroy();
        tokenBuffer.destroy();
        outputBuffer.destroy();
        readBuffer.destroy();
        
        return maxIdx;
    }

    decodeToken(tokenId) {
        const fakeTokens = ["the", "a", "an", "model", "language", "neural", "network", "deep", "learning", "AI", 
                           "is", "are", "was", "were", "be", "to", "of", "and", "in", "that"];
        if (tokenId < fakeTokens.length) {
            return " " + fakeTokens[tokenId];
        }
        return ` [${tokenId}]`;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async runInference(prompt, onToken) {
        const jumpstartData = await this.jumpstart(prompt);
        
        for (let i = 0; i < jumpstartData.initialTokens.length; i++) {
            onToken(jumpstartData.initialTokens[i], i === 0);
            await this.delay(50);
        }
        
        const startLayer = jumpstartData.nextLayer || 1;
        console.log(`Inference: Processing shards ${startLayer} to ${this.totalShards - 1}`);
        
        for (let layer = startLayer; layer < this.totalShards; layer++) {
            const shardData = await this.fetchShard(layer);
            const layerData = new Float32Array(shardData);
            
            const nextTokenId = await this.processLayerWithGPU(layerData, []);
            const nextToken = this.decodeToken(nextTokenId);
            
            onToken(nextToken, false);
            await this.delay(30);
        }
        
        onToken("\n\n[Streaming complete]", false);
        console.log("Inference complete");
    }
    
    destroy() {
        for (const buffer of this.activeBuffers.values()) {
            buffer.destroy();
        }
        this.activeBuffers.clear();
    }
}

export { StreamWeightManager };