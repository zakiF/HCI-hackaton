# Git Workflow for Hackathon

## Branch Strategy

Each person works on their own feature branch, then we integrate at Day 2 11am.

---

## 📦 Branch Assignments

| Person | Branch Name | Module File |
|--------|-------------|-------------|
| Zaki + Udhaya | `feature/multi-gene` | `modules/R/multi_gene_viz.R` |
| Qing Li | `feature/filters` | `modules/python/llm_filters.py` |
| Miao Ai | `feature/conversation` | `modules/python/conversation.py` |
| Tayler Fearn | `feature/stats` | `modules/python/llm_stats.py` |
| David Stone | `feature/rag` | `modules/python/llm_rag.py` |

---

## 🚀 Setup (Before Hackathon Starts)

### **For Zaki (you)**: Create branches

```bash
cd /Users/zakiwilmot/Documents/GitHub/HCI-hackaton

# Create and push feature branches
git checkout -b feature/multi-gene
git push -u origin feature/multi-gene

git checkout main
git checkout -b feature/filters
git push -u origin feature/filters

git checkout main
git checkout -b feature/conversation
git push -u origin feature/conversation

git checkout main
git checkout -b feature/stats
git push -u origin feature/stats

git checkout main
git checkout -b feature/rag
git push -u origin feature/rag

# Return to main
git checkout main
```

---

## 👥 For Each Team Member (Day 1 10am)

### **Step 1: Clone Repository**

```bash
# Clone if you haven't already
git clone https://github.com/YOUR-USERNAME/HCI-hackaton.git
cd HCI-hackaton
```

### **Step 2: Checkout Your Branch**

**Zaki + Udhaya**:
```bash
git checkout feature/multi-gene
```

**Qing**:
```bash
git checkout feature/filters
```

**Miao**:
```bash
git checkout feature/conversation
```

**Tayler**:
```bash
git checkout feature/stats
```

**David**:
```bash
git checkout feature/rag
```

---

## 💾 During Development (Day 1-2)

### **Commit Often**

Save your work frequently:

```bash
# Check what you've changed
git status

# Add your changes
git add modules/python/your_module.py

# Commit with descriptive message
git commit -m "Implement extract_filters function with LLM"

# Push to your branch (safe - won't affect others)
git push origin feature/YOUR-FEATURE
```

### **Good Commit Messages**

✅ Good:
- "Add LLM prompt for gene extraction"
- "Fix filter extraction for edge cases"
- "Implement conversation context resolution"

❌ Bad:
- "Update code"
- "Fix stuff"
- "WIP"

---

## 🔀 Integration (Day 2 11am)

### **Option 1: Manual Integration** (Recommended)

**One person** (Tayler or David) drives this process:

```bash
# Start fresh from main
git checkout main
git pull origin main

# Create integration branch
git checkout -b integration

# Manually copy each feature:

# 1. David's RAG feature
git checkout feature/rag -- modules/python/llm_rag.py
# Edit chatbot.py to add David's import and route
# Test - does it work?

# 2. Tayler's stats feature
git checkout feature/stats -- modules/python/llm_stats.py
# Edit chatbot.py to add Tayler's import and route
# Test - does it work?

# ... continue for each feature

# Commit integrated version
git add .
git commit -m "Integrate all LLM features"
git push origin integration
```

### **Option 2: Sequential Merges** (If team comfortable with Git)

```bash
git checkout main

# Merge one feature at a time
git merge feature/rag
# TEST - does everything still work?

git merge feature/stats
# TEST again

git merge feature/filters
# TEST again

# ... etc
```

---

## 🚨 Handling Conflicts

### **If you get merge conflicts on `chatbot.py`**:

**Don't panic!** This is why we have feature flags.

**Solution**:
1. Open `chatbot.py`
2. Find conflict markers: `<<<<<<`, `======`, `>>>>>>`
3. Keep both changes (both imports, both routes)
4. Test that both features work
5. Commit resolved version

**Example conflict**:
```python
<<<<<<< HEAD
from modules.python.llm_rag import answer_gene_question
=======
from modules.python.llm_stats import handle_stats_query
>>>>>>> feature/stats
```

**Resolution** (keep both):
```python
from modules.python.llm_rag import answer_gene_question
from modules.python.llm_stats import handle_stats_query
```

---

## 📁 What Files Can You Edit?

### **Safe to Edit (No Conflicts)**:

Each person should **primarily** edit their own module file:

- Zaki+Udhaya: `modules/R/multi_gene_viz.R` ✅
- Qing: `modules/python/llm_filters.py` ✅
- Miao: `modules/python/conversation.py` ✅
- Tayler: `modules/python/llm_stats.py` ✅
- David: `modules/python/llm_rag.py` ✅

**These files won't conflict** because each person works on different files!

### **Risky to Edit (Might Conflict)**:

- `app/python/chatbot.py` ⚠️ (everyone needs to add to this)
  - **Solution**: Let integration person handle this on Day 2

### **Don't Edit**:

- `app/python/chatbot_base.py` ❌ (backup - don't touch)
- `utils/python/ollama_utils.py` ❌ (shared - already complete)
- `utils/R/ollama_utils.R` ❌ (shared - already complete)

---

## 🎯 Best Practices

### **During Hackathon (Day 1-2)**:

1. **Work only on your branch**
2. **Commit frequently** (every 30-60 min)
3. **Push to remote** regularly (backup!)
4. **Don't merge others' code** yet
5. **Focus on your module file**

### **Before Pushing**:

```bash
# Always check what you're about to commit
git status
git diff

# Make sure you're on YOUR branch
git branch
# Should show: * feature/YOUR-FEATURE

# Then push
git push origin feature/YOUR-FEATURE
```

### **If You Make a Mistake**:

```bash
# Undo last commit (keeps changes)
git reset HEAD~1

# Discard all changes (DANGER - can't undo!)
git reset --hard HEAD

# Switch to different branch without committing
git stash         # Save changes temporarily
git checkout main
git checkout feature/YOUR-FEATURE
git stash pop     # Restore changes
```

---

## 📊 Git Commands Cheat Sheet

### **Daily Commands**:

```bash
# See what you've changed
git status

# See detailed changes
git diff

# Stage changes
git add your_file.py

# Commit
git commit -m "Your message"

# Push to remote
git push origin feature/YOUR-FEATURE

# Pull latest from your branch
git pull origin feature/YOUR-FEATURE
```

### **Checking State**:

```bash
# Which branch am I on?
git branch

# What's the commit history?
git log --oneline

# Did I commit this file?
git log -- your_file.py
```

### **Emergency Commands**:

```bash
# I'm on the wrong branch!
git checkout feature/YOUR-FEATURE

# I committed to wrong branch!
git cherry-pick COMMIT-HASH  # On correct branch

# I need to start over!
git checkout main
git pull origin main
git checkout feature/YOUR-FEATURE
git reset --hard origin/feature/YOUR-FEATURE
```

---

## 🔍 Troubleshooting

### **"I can't push - rejected"**

**Problem**: Someone else pushed to your branch

**Solution**:
```bash
git pull origin feature/YOUR-FEATURE
# Resolve any conflicts
git push origin feature/YOUR-FEATURE
```

### **"I have uncommitted changes but need to switch branches"**

**Solution**:
```bash
git stash                    # Save changes
git checkout other-branch    # Switch
git checkout feature/YOUR-FEATURE  # Switch back
git stash pop                # Restore changes
```

### **"I accidentally edited chatbot.py on my branch"**

**Solution**: That's okay! Just commit it. Integration person will handle merging all edits.

```bash
git add app/python/chatbot.py
git commit -m "Add my feature to chatbot"
git push origin feature/YOUR-FEATURE
```

---

## 🎬 Integration Day Workflow (Day 2 11am)

### **For the Integration Person** (Tayler or David):

**Timeline**:

**11:00-11:10**: Setup
```bash
git checkout main
git pull
git checkout -b integration
```

**11:10-11:40**: Add features one by one
```bash
# Copy each module file
git checkout feature/rag -- modules/python/llm_rag.py

# Manually edit chatbot.py to add route

# Test
streamlit run app/python/chatbot.py

# If works, commit. If breaks, disable feature flag.
git add .
git commit -m "Add RAG feature"
```

**11:40-11:45**: Final test
```bash
# Test all features together
streamlit run app/python/chatbot.py
```

**11:45**: Merge to main
```bash
git checkout main
git merge integration
git push origin main
```

### **For Everyone Else**:

- Stand by to explain your code
- Don't interrupt the integration person
- Fix bugs in your module if found
- Test your feature standalone as backup

---

## ✅ Pre-Hackathon Checklist

**For Zaki (you)**:
- [ ] Create all feature branches
- [ ] Push branches to remote
- [ ] Test that team can clone and checkout

**For Each Team Member**:
- [ ] Clone repository
- [ ] Checkout your feature branch
- [ ] Verify you can commit and push

**Test on Day 1 10am**:
```bash
# Everyone try this:
echo "test" >> test.txt
git add test.txt
git commit -m "Test commit"
git push origin feature/YOUR-FEATURE
# If this works, you're ready!
```

---

## 📝 Branch Merge Order (Suggested)

If integrating sequentially:

1. **Base** (main) - already working
2. **RAG** (David) - standalone, won't interfere
3. **Stats** (Tayler) - standalone, won't interfere
4. **Filters** (Qing) - modifies data flow
5. **Conversation** (Miao) - modifies input processing
6. **Multi-gene** (Zaki+Udhaya) - might need R bridge

**Rationale**: Add standalone features first, then features that modify the pipeline.

---

## 🎯 Remember

**The goal is LEARNING, not perfect Git workflow!**

- If you make Git mistakes → that's okay, we'll fix them
- If conflicts happen → we have feature flags as backup
- If integration fails → we demo features separately

**Git is a tool to help us collaborate, not a barrier to learning LLMs!**
