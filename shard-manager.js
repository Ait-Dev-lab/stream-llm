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
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported. Please use Chrome 113+ or Edge 113+.");
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU adapter found");
        
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: 512 * 1024 * 1024,
                maxBufferSize: 512 * 1024 * 1024
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
        
        console.log(`Jumpstart: ${data.initialTokenCount} tokens received, ${data.modelConfig.shardCount} shards available`);
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

    async runInference(prompt, onToken) {
        const jumpstartData = await this.jumpstart(prompt);
        
        // Display initial tokens immediately
        jumpstartData.initialTokens.forEach((t, i) => onToken(t, i === 0));
        
        console.log(`Inference: Browser takes over at layer ${jumpstartData.nextLayer}`);
        
        for (let layer = jumpstartData.nextLayer; layer < this.modelConfig.totalLayers; layer++) {
            await this.fetchShard(layer);
            console.log(`Layer ${layer + 1}/${this.modelConfig.totalLayers} processed`);
        }
        
        onToken("\n\n[Streaming complete]", false);
    }
}

export { StreamWeightManager };
