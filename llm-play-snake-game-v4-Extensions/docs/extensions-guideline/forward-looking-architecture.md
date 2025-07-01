## 🚀 **Forward-Looking Architecture**

### **No Backward Compatibility**
We refactor with a future-proof mindset, creating fresh, self-consistent, and self-contained systems. We do not maintain backward compatibility - deprecated code should be removed entirely. No legacy considerations for extensions.

For Task-0, maintain output schema compliance as defined in the reference files above.

### **No Code Pollution**
No pollution from extensions (Task 1-5) into the ROOT folder. Extension-specific terminology (e.g., "heuristics", "reinforcement learning") should not appear in the ROOT folder - only in the `extensions/` folder.

### **No Over-Preparation**
Let future tasks implement their own required code/functions. Avoid overkill and over-preparation in the ROOT folder.


