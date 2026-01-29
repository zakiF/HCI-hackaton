# Ollama Setup Guide

## 🎯 Goal

Install and configure Ollama so you can run local LLMs for the hackathon.

**Do this BEFORE the hackathon starts!**

---

## Step 1: Install Ollama

### **macOS**

1. Visit: https://ollama.com/download
2. Download the macOS installer
3. Open the `.dmg` file and drag Ollama to Applications
4. Open Ollama from Applications

**Or via Homebrew**:
```bash
brew install ollama
```

### **Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### **Windows**

1. Visit: https://ollama.com/download
2. Download Windows installer
3. Run the installer
4. Follow installation prompts

---

## Step 2: Download the Model

We're all using **llama3.2** for consistency.

```bash
ollama pull llama3.2
```

**Expected output**:
```
pulling manifest
pulling 8eeb52dfb3bb... 100% ▕████████████▏ 2.0 GB
pulling 73b313b5552d... 100% ▕████████████▏  11 KB
...
success
```

**Note**: This downloads ~2-4 GB. Do this on good WiFi!

---

## Step 3: Start Ollama Server

### **Option A: Start from Terminal** (Recommended)

```bash
ollama serve
```

**Expected output**:
```
Couldn't find '/Users/your-name/.ollama/id_ed25519'. Generating new private key.
Your new public key is:

ssh-ed25519 AAAAC3...

2024/01/28 10:00:00 routes.go:1000: Listening on 127.0.0.1:11434
```

**Keep this terminal window open!** Ollama must be running for the chatbot to work.

### **Option B: Run as Background Service**

**macOS**:
```bash
brew services start ollama
```

**Linux**:
```bash
systemctl start ollama
```

---

## Step 4: Test Ollama

### **Test 1: Basic Command**

In a NEW terminal window (keep `ollama serve` running in the first):

```bash
ollama run llama3.2 "What is 2+2?"
```

**Expected output**:
```
The answer is 4!
```

### **Test 2: Python Test**

```python
import requests

url = "http://localhost:11434/api/generate"
data = {
    "model": "llama3.2",
    "prompt": "What is the capital of France?",
    "stream": False
}

response = requests.post(url, json=data)
print(response.json()['response'])
```

**Expected output**: `"The capital of France is Paris."`

### **Test 3: R Test**

```r
library(httr)
library(jsonlite)

url <- "http://localhost:11434/api/generate"

data <- list(
  model = "llama3.2",
  prompt = "What is the capital of France?",
  stream = FALSE
)

response <- POST(
  url,
  body = toJSON(data, auto_unbox = TRUE),
  encode = "json",
  content_type_json()
)

result <- content(response, as = "parsed")
cat(result$response)
```

**Expected output**: `"The capital of France is Paris."`

---

## Step 5: Install Required Packages

### **Python Packages**

```bash
pip install streamlit pandas matplotlib requests scipy
```

**Or with conda**:
```bash
conda install streamlit pandas matplotlib requests scipy
```

### **R Packages**

```r
install.packages(c("shiny", "httr", "jsonlite", "tidyverse", "ggplot2"))
```

---

## 🔍 Troubleshooting

### **"Can't connect to Ollama"**

**Symptom**: `Connection refused` error

**Solution**:
1. Make sure Ollama is running: `ollama serve`
2. Check it's on correct port: `curl http://localhost:11434/api/tags`
3. Try restarting Ollama

### **"Model not found"**

**Symptom**: `model 'llama3.2' not found`

**Solution**:
```bash
ollama list  # Check installed models
ollama pull llama3.2  # Download if missing
```

### **"Ollama is slow / hanging"**

**Symptom**: Requests take >1 minute

**Possible causes**:
1. First request is slow (model loads into memory)
2. Your computer has limited RAM (llama3.2 needs ~4GB)
3. Other programs using too much memory

**Solutions**:
- Wait for first request to complete (subsequent requests are faster)
- Close other applications to free RAM
- Use smaller model: `ollama pull llama3.2:3b` (lighter version)

### **"Permission denied"**

**Linux**:
```bash
sudo systemctl start ollama
```

**macOS**: Try reinstalling Ollama

---

## 📊 Verifying Installation

Run this checklist:

```bash
# 1. Ollama is installed
ollama --version
# Should show: ollama version 0.X.X

# 2. Model is downloaded
ollama list
# Should show: llama3.2

# 3. Server is running
curl http://localhost:11434/api/tags
# Should return JSON with model list

# 4. Python can connect
python -c "import requests; print(requests.get('http://localhost:11434/api/tags').status_code)"
# Should print: 200

# 5. R can connect (in R console)
# httr::GET("http://localhost:11434/api/tags")
# Should show: Response [200]
```

**All checks passed?** ✅ You're ready for the hackathon!

---

## 🎯 Day-of Hackathon Startup

On Day 1 at 9:45am (before 10am start):

### **Quick Start Checklist**:

```bash
# 1. Start Ollama (in one terminal)
ollama serve

# 2. Verify it's running (in another terminal)
curl http://localhost:11434/api/tags

# 3. Navigate to repo
cd /path/to/HCI-hackaton

# 4. Checkout your branch
git checkout feature/YOUR-FEATURE

# 5. Test chatbot (just to verify)
streamlit run app/python/chatbot_base.py
```

**You're ready to go!** 🚀

---

## 💡 Tips

### **Keep Ollama Running**

Don't close the terminal with `ollama serve`! If you do, LLM calls will fail.

### **Monitor Ollama**

Watch the `ollama serve` terminal to see LLM requests happening in real-time.

### **Restart if Needed**

If Ollama becomes unresponsive:
```bash
# Stop
Ctrl+C (in ollama serve terminal)

# Start again
ollama serve
```

---

## 🔧 Advanced Configuration (Optional)

### **Change Port** (if 11434 is taken)

```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

Then update `OLLAMA_URL` in:
- `utils/python/ollama_utils.py`
- `utils/R/ollama_utils.R`

### **Use Different Model**

If you want to try a different model:

```bash
ollama pull llama3.1        # Larger, more capable
ollama pull llama3.2:3b     # Smaller, faster
ollama pull deepseek-r1:7b  # Better reasoning
ollama pull qwen2.5-coder   # Better code generation
```

Then change `DEFAULT_MODEL` in utils files.

**But for hackathon**: Stick with `llama3.2` for consistency!

---

## 📚 Resources

- **Ollama Docs**: https://ollama.com/docs
- **Model Library**: https://ollama.com/library
- **GitHub**: https://github.com/ollama/ollama
- **Discord**: https://discord.gg/ollama

---

## ⚠️ Important Notes

### **Offline Use**

Once you've downloaded llama3.2, Ollama works **completely offline**.

Great for working on secure/air-gapped systems!

### **Privacy**

All processing happens **locally on your machine**. No data sent to cloud.

### **Resource Usage**

- **RAM**: ~4-8 GB while model is loaded
- **Disk**: ~2-4 GB for llama3.2 model
- **CPU/GPU**: Uses all available (faster on Apple Silicon / NVIDIA GPUs)

---

## ✅ Pre-Hackathon Checklist

**Complete these before Day 1**:

- [ ] Ollama installed and `ollama --version` works
- [ ] llama3.2 model downloaded (`ollama list` shows it)
- [ ] Can start server (`ollama serve` works)
- [ ] Can make test request (`ollama run llama3.2 "test"` works)
- [ ] Python requests library works with Ollama
- [ ] R httr library works with Ollama (for Zaki/Udhaya)
- [ ] All required packages installed (streamlit, scipy, etc.)

**If ANY of these fail**, ask for help in Slack **before** Day 1!

---

## 🆘 Getting Help

**Before hackathon**:
- Post in Slack with error messages
- Share output of `ollama --version` and `ollama list`

**During hackathon**:
- David and Tayler can help debug Ollama issues
- Check the `ollama serve` terminal for error messages

---

## 🎉 You're Ready!

Once you've completed this guide, you're all set for the hackathon.

**See you at 10am on Day 1!** 🚀
