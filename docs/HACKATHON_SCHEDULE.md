# Hackathon Schedule (7 Hours Total)

## Time Available

| Day | Session | Duration | Cumulative |
|-----|---------|----------|------------|
| **Day 1** | 10am-12pm | 2 hours | 2 hours |
| **Day 1** | 3pm-5pm | 2 hours | 4 hours |
| **Day 2** | 9am-12pm | 3 hours | 7 hours |
| **Day 2** | 2pm-3pm | 1 hour | 8 hours |
| **Day 2** | **3pm** | **Presentation** | - |

**Total working time: 7-8 hours**

---

## 📅 Day 1 Morning: 10am-12pm (2 hours)

### **10:00-10:30 - LLM Workshop** (All Together)

**Led by**: David Stone + Tayler Fearn

**Topics**:
1. Quick intro: What are local LLMs? Why Ollama?
2. **Live demo**: Making your first LLM call
3. Key concepts:
   - Prompts (what you ask)
   - Temperature (creativity slider)
   - Structured output (getting JSON)
4. Common pitfalls and how to avoid them

**Hands-on**:
- Everyone runs their first `call_ollama()` command
- Test basic extraction: "Extract gene name from: Show me TP53"

**Goal**: Everyone successfully calls Ollama at least once

---

### **10:30-10:45 - Task Assignment & Questions** (All Together)

**Activities**:
1. Review task assignments (who's doing what)
2. Clarify questions about:
   - Module structure
   - Expected deliverables
   - Git workflow
3. Quick check: Does everyone have:
   - Ollama running? ✓
   - Correct branch checked out? ✓
   - Module file open? ✓

**Assignments Recap**:
- Zaki + Udhaya → Multi-gene viz (R)
- Qing → Filtering (Python)
- Miao → Conversation (Python)
- Tayler → Stats (Python)
- David → RAG (Python)

---

### **10:45-12:00 - Start Individual Work** (Independent)

**Goal for this session**: Make your first successful LLM call for your specific task

**What everyone should accomplish**:

**Zaki + Udhaya**:
- Test `call_ollama()` from R
- Try extracting a gene name: `extract_gene_names("Show me TP53")`
- Goal: Return `c("TP53")`

**Qing**:
- Test `call_ollama()` from Python
- Try extracting filter: `extract_filters("Show TP53 in tumor only")`
- Goal: Return filter dict

**Miao**:
- Test conversation context detection
- Goal: Detect "it" as a follow-up reference

**Tayler**:
- Test intent classification
- Goal: Classify "Is TP53 significant?" as stats query

**David**:
- Test gene info retrieval
- Goal: Retrieve TP53 info from database

**Support**: David + Tayler float between people to help with issues

**By 12pm**: Everyone has called LLM successfully at least once

---

## 🍕 Lunch Break: 12pm-3pm

**Pre-work for afternoon**:
- Think about: What prompt wording worked? What didn't?
- Optional: Continue working if you're on a roll!

---

## 📅 Day 1 Afternoon: 3pm-5pm (2 hours)

### **3:00-3:15 - Quick Sync** (Optional, All Together)

**Questions**:
- Any blockers from morning session?
- Anyone want to share an interesting LLM behavior?
- Quick tip sharing

**Keep it short** - more time for coding!

---

### **3:15-4:45 - Focused Development** (Independent)

**Goal**: Get your core feature ~50% working

**Milestones by person**:

**Zaki + Udhaya**:
- ✅ `extract_gene_names()` working for 1-3 genes
- ✅ `detect_plot_type()` basic implementation
- 🔄 Start on one plotting function (boxplot or violin)

**Qing**:
- ✅ `extract_filters()` working for common cases
- ✅ `apply_filters()` subset working
- 🔄 Test with different phrasings

**Miao**:
- ✅ `_is_followup_query()` working
- ✅ `_resolve_with_llm()` basic implementation
- 🔄 Test conversation flow

**Tayler**:
- ✅ `is_stats_query()` classification working
- ✅ `extract_stats_parameters()` for t-test case
- 🔄 Test `run_ttest()` function

**David**:
- ✅ `is_gene_question()` detection working
- ✅ Expanded gene database (5-10 genes)
- ✅ `generate_gene_explanation()` basic version

**Strategy**:
- Focus on getting one thing fully working
- Test frequently with different inputs
- Iterate on prompts if LLM fails

---

### **4:45-5:00 - Day 1 Wrap-up** (All Together)

**Quick round-robin** (1 min per person):
- What's working?
- What's not working yet?
- What will you focus on tomorrow?

**Plan for Day 2**:
- Finish features in morning
- Integrate at 11am
- Polish for demo at 3pm

**Homework** (optional):
- Think about test cases for your feature
- Refine your prompts if they're not working well

---

## 🌙 Evening: Optional

Some people might want to continue working - that's fine!
But not required. Get rest for Day 2!

---

## 📅 Day 2 Morning: 9am-12pm (3 hours) - CRITICAL SESSION

### **9:00-9:15 - Integration Planning** (All Together)

**Decisions**:
1. Who's driving integration? (Suggest: Tayler or David)
2. What's the merge order?
3. Backup plan if integration fails?

**Integration Strategy**:
- Integration person will manually combine features
- Use feature flags to toggle on/off
- Test each feature individually before combining

**Priority order** (if time runs short):
1. Base chatbot (must work)
2. Zaki+Udhaya multi-gene (visual impact)
3. David RAG (impressive demo)
4. Tayler stats (good demo)
5. Qing filters (nice to have)
6. Miao conversation (nice to have)

---

### **9:15-11:00 - Finish Individual Features** (Independent)

**Goal**: Your feature is 100% working on your branch

**Final tasks by person**:

**Zaki + Udhaya**:
- Complete all 3 plotting functions
- Test with multiple queries
- Handle edge cases (invalid genes, etc.)

**Qing**:
- Refine filter extraction for edge cases
- Test with various phrasings
- Handle "no filter" case

**Miao**:
- Test multi-turn conversations
- Handle ambiguous references
- Reset conversation function

**Tayler**:
- Implement ANOVA (3+ groups)
- Test summary generation
- Handle edge cases

**David**:
- Add more genes to database (10-15 total)
- Refine explanation generation
- Test various question formats

**Testing checklist**:
- ✅ Works with correct input
- ✅ Handles typos gracefully
- ✅ Handles missing genes
- ✅ Returns sensible defaults if LLM fails

---

### **11:00-11:45 - Integration Sprint** (One Person Drives, Others Support)

**Integrator** (Tayler or David):

**11:00-11:15**: Test base chatbot still works

**11:15-11:45**: Add features one by one:

1. Add David's RAG (15 min)
   - Import functions
   - Add route in chatbot.py
   - Test: "What does TP53 do?"
   - If breaks → disable, move on

2. Add Tayler's stats (15 min)
   - Import functions
   - Add route
   - Test: "Is TP53 significant?"
   - If breaks → disable, move on

3. Add Qing's filters (if time)
4. Add Miao's conversation (if time)
5. Add Zaki+Udhaya's multi-gene (if time, might be complex)

**Everyone else**:
- Stand by to explain your code
- Fix quick bugs if found
- Don't interrupt the integrator!

---

### **11:45-12:00 - Integration Testing** (All Together)

**Run through demo scenarios**:
1. Basic: "Show me TP53"
2. RAG: "What does TP53 do?"
3. Stats: "Is TP53 significant?"
4. Filters: "Show TP53 in tumor only" (if integrated)
5. Multi-gene: "Compare TP53 and BRCA1" (if integrated)

**Fix critical bugs only**

**Take screenshots** of working features (backup if demo fails!)

---

## 🍕 Lunch: 12pm-2pm

**Relax!** The hard work is done.

Optional: Practice demo transitions

---

## 📅 Day 2 Afternoon: 2pm-3pm (1 hour) - Final Polish

### **2:00-2:15 - Demo Script Planning**

**Decide**:
1. What order to show features?
2. Who explains what? (Each person explains their own feature)
3. Prepare 2-3 example queries per feature

**Suggested demo flow**:
1. Intro (Zaki, 1 min): "We built an LLM chatbot to learn local LLMs"
2. Live demo (5 min):
   - Base feature
   - Each working feature (30 sec each)
3. Behind the scenes (2 min):
   - Show one LLM prompt/response
   - Explain what we learned
4. Q&A (2 min)

---

### **2:15-2:45 - Demo Rehearsal**

**Run through presentation**:
- Time it (should be 10-12 minutes)
- Make sure all features work
- Prepare backup (show feature standalone if integration broke)

**Create backup materials**:
- Screenshots of working features
- Screen recording of successful run (optional)

---

### **2:45-3:00 - Final Fixes & Breathing Room**

- Fix any last-minute issues
- Test one more time
- Deep breath!

---

## 🎤 3:00pm - PRESENTATION

### **Presentation Structure** (15 minutes total)

**1. Introduction** (Zaki, 2 min)
- Goal: Learn local LLMs hands-on
- Team composition
- What we built

**2. Live Demo** (8 min)
- Start simple: "Show me TP53" (base feature)
- Each person demos their feature:
  - Zaki+Udhaya: Multi-gene heatmap
  - Qing: Filtering
  - Miao: Follow-up conversation
  - Tayler: Statistical test
  - David: Gene information
- Show integrated chatbot in action

**3. Behind the Scenes** (3 min)
- Pick ONE feature (e.g., Tayler's stats)
- Show the LLM prompt
- Show the LLM response
- Show how we parsed it
- "This is what we learned about prompt engineering"

**4. Lessons Learned** (2 min)
- What surprised us about LLMs?
- What would we do differently?
- How we'll use this in our research?
- Challenges we overcame

**5. Q&A** (flexible)

---

## 🎯 Success Criteria

By end of hackathon, we should have:

### **Technical Deliverables**:
- ✅ Working chatbot (at minimum, base version)
- ✅ 3-5 LLM features (even if not all integrated)
- ✅ Each person's module works standalone

### **Learning Deliverables**:
Each person can answer:
- ✅ How do I call a local LLM?
- ✅ How do I write prompts for extraction/classification?
- ✅ What are LLM limitations?
- ✅ Where would I use LLMs in my work?

### **Demo Deliverables**:
- ✅ 10-15 minute presentation
- ✅ Live demo of working features
- ✅ Explanation of LLM concepts learned

---

## ⏰ Time Management Tips

### **If Running Behind**:
- **Hour 4**: Focus on getting ONE thing fully working, not everything partially
- **Hour 6**: Accept what works, move to integration
- **Hour 7**: Feature flags save you - disable what's broken

### **If Ahead of Schedule**:
- **Add edge case handling**
- **Improve LLM prompts for better accuracy**
- **Add more test cases**
- **Polish the UI**

### **Emergency Fallback**:
If integration completely fails at 11am:
- Each person demos their feature **standalone**
- Run module test files directly
- Show LLM prompts/responses
- **This still counts as success** - learning happened!

---

## 📊 Milestones Checklist

### **Day 1 End**:
- [ ] Everyone has called Ollama successfully
- [ ] Everyone has basic feature ~50% working
- [ ] Everyone knows what they're focusing on Day 2

### **Day 2 11am**:
- [ ] Everyone's feature works standalone
- [ ] Integration started
- [ ] Base chatbot still works

### **Day 2 2pm**:
- [ ] Demo script ready
- [ ] At least 3 features working (base + 2 others)
- [ ] Backup plan if live demo fails

### **Day 2 3pm**:
- [ ] Successful presentation!
- [ ] Everyone learned LLM skills
- [ ] We have a working prototype

---

## 🎉 Post-Hackathon

**Celebrate!** You built an LLM-powered application in 7 hours!

**Optional next steps**:
- Continue developing features
- Clean up code
- Write blog post about learnings
- Apply LLM skills to your actual research

**Key takeaway**: You now know how to work with local LLMs!
