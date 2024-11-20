# Advanced Agentic AI Chatbot

## Revolutionary AI Assistant with RAG, Memory, and Tool Registration

This is a state-of-the-art agentic AI chatbot that represents the next evolution in conversational AI systems.

## 🚀 Revolutionary Features

### 🧠 RAG-Enhanced Intelligence
- **Database Schema RAG**: Uses documentation to understand your database structure
- **Intelligent SQL Generation**: Creates accurate queries based on schema knowledge
- **Context-Aware Responses**: Leverages documentation for better understanding

### 🧮 Advanced Memory Management  
- **Session Memory**: Automatically caches query results for follow-up questions
- **Conversation Continuity**: Remembers previous data across the conversation
- **Smart Context Detection**: Knows when to use cached data vs fetch new data

### ⚙️ Single LLM Call Architecture
- **Tool Registration**: Registers multiple tools (SQL, Python, Visualization) with the LLM
- **Efficient Processing**: Handles complex requests in a single LLM call
- **Parallel Execution**: Can execute multiple tools simultaneously

### 🐍 Python Code Generation & Execution
- **Dynamic Analysis**: Generates Python code for statistical analysis
- **Safe Execution**: Runs code in controlled environment
- **Advanced Statistics**: Correlation, regression, ANOVA, and more

### 📊 Intelligent Visualizations
- **Context-Aware Charts**: Creates appropriate visualizations based on data type
- **Multiple Chart Types**: Bar, line, scatter, histogram, pie, heatmap
- **Interactive Plotly**: Fully interactive charts with zoom, pan, hover

### 💬 Conversation Intelligence
- **Follow-up Detection**: Understands when questions relate to previous data
- **Memory Integration**: Seamlessly uses cached data for follow-up questions
- **Context Preservation**: Maintains conversation flow across complex interactions

## 🎯 What Makes This Advanced

### Traditional Chatbots:
- Static SQL templates
- No memory between queries  
- Multiple LLM calls for complex tasks
- Basic chart generation
- No conversation continuity

### This Advanced System:
- **Dynamic RAG-powered SQL generation**
- **Intelligent memory management** 
- **Single LLM call with tool registration**
- **Python code generation for analysis**
- **Smart conversation continuity**

## 🛠 Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Ollama:**
   ```bash
   # Install Ollama from https://ollama.com
   ollama pull qwen3:8b
   ollama serve
   ```

3. **Run the advanced chatbot:**
   ```bash
   # Linux/macOS
   ./run_advanced.sh
   
   # Windows  
   run_advanced.bat
   
   # Or manually
   streamlit run main_advanced.py
   ```

## 💡 Example Advanced Interactions

### Smart Memory Usage:
```
You: "Show me employee data"
AI: [Fetches data, stores in memory] "Here are the employees... [shows data]"

You: "What's the average salary?" 
AI: [Uses cached data] "Based on the employee data we just fetched, the average salary is $67,500"

You: "Create a salary distribution chart"
AI: [Uses cached data + generates visualization] [Shows interactive chart]
```

### Advanced Analysis:
```
You: "Analyze the correlation between age and salary"
AI: [Generates Python code, executes analysis] 
    "I found a moderate positive correlation (r=0.43, p<0.05) between age and salary..."
    [Shows scatter plot with regression line]
```

### Comprehensive Reports:
```
You: "Create a comprehensive sales analysis dashboard"
AI: [Single LLM call with multiple tools]
    - Fetches sales data
    - Generates statistical analysis
    - Creates multiple visualizations  
    - Provides detailed insights
    [Shows complete dashboard with multiple charts and analysis]
```

## 🔧 Advanced Configuration

The system supports extensive configuration through `config.py`:

- **Memory Settings**: Control cache size and retention
- **Tool Configuration**: Enable/disable specific capabilities
- **Safety Settings**: Control code execution permissions
- **Performance Tuning**: Adjust timeouts and limits

## 🧪 Technical Innovation

### RAG Integration:
- Loads database schemas into vector embeddings
- Uses semantic search for relevant schema information
- Provides context-aware SQL generation

### Memory Architecture:
- Session-based caching with metadata
- Intelligent cache invalidation
- Memory usage optimization

### Tool Registration:
- Qwen3-compatible tool definitions
- Parallel tool execution capability
- Error handling and recovery

### Code Generation:
- Safe Python code execution
- Comprehensive analysis libraries
- Result validation and formatting

## 🎉 Ready for Production

This advanced system is production-ready with:
- Comprehensive error handling
- Security safeguards
- Performance optimization
- Extensive documentation
- Modular architecture

Experience the future of conversational AI today!
