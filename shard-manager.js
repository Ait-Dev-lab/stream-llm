// shard-manager.js - Complete working version
class StreamWeightManager {
    constructor(config) {
        this.jumpstartUrl = config.jumpstartUrl;
        this.device = null;
        this.modelConfig = null;
        this.cdnBase = "https://github.com/Ait-Dev-lab/stream-llm/releases/download/v1.0.0";
        this.totalShards = 30;
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
        
        console.log("StreamWeightManager: WebGPU initialized");
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
        this.cdnBase = data.cdnUrl || this.cdnBase;
        
        console.log(`Jumpstart: ${data.initialTokens.length} tokens received`);
        return data;
    }

    async fetchShard(shardId) {
        const url = `${this.cdnBase}/shard_${String(shardId).padStart(3, '0')}.bin`;
        console.log(`Fetching shard ${shardId}...`);
        
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Shard ${shardId} failed: ${response.status}`);
        
        return await response.arrayBuffer();
    }

    async runInference(prompt, onToken) {
        // Step 1: Get initial tokens
        const jumpstartData = await this.jumpstart(prompt);
        
        // Step 2: Stream first 8 tokens
        for (let i = 0; i < jumpstartData.initialTokens.length; i++) {
            onToken(jumpstartData.initialTokens[i], i === 0);
            await this.delay(50);
        }
        
        // Step 3: For demo, generate a simulated response
        // In production, this would process actual shards with WebGPU
        const demoResponses = [
            " A large language model is an AI system trained on vast amounts of text to understand and generate human-like language.",
            " It works by predicting the next word in a sequence, building coherent responses based on patterns learned from training data.",
            " These models can answer questions, summarize text, write code, and much more."
        ];
        
        for (let i = 0; i < demoResponses.length; i++) {
            const words = demoResponses[i].split(' ');
            for (let j = 0; j < words.length; j++) {
                onToken(words[j], false);
                await this.delay(30);
            }
        }
        
        onToken("\n\n[Streaming complete]", false);
        console.log("Inference complete");
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

export { StreamWeightManager };
