## 🚫 **CRITICAL: NO singleton_utils.py in extensions/common/**

**STOP! READ THIS FIRST**: This project **explicitly FORBIDS**:
- ❌ **singleton_utils.py in extensions/common/utils/**
- ❌ **Any wrapper around ROOT/utils/singleton_utils.py**
- ✅ Instead, we must **USE ROOT/utils/singleton_utils.py** directly when truly needed (it's already generic)


