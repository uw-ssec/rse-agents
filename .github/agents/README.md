# RSE Agents Directory

This directory contains custom agent configurations for Research Software Engineering (RSE) and scientific computing tasks. Each agent is designed to assist with specific aspects of scientific software development.

## 📂 Directory Contents

Agent files in this directory are Markdown files that define specialized AI assistants for scientific computing tasks. Each file contains:

- **Agent Description**: What the agent does and its areas of expertise
- **Usage Guidelines**: When and how to use the agent
- **Agent Prompt**: The instruction text that defines the agent's behavior

## 🔨 Creating New Agents

When creating a new agent for this directory:

### 1. Choose an Appropriate Name

- Use kebab-case: `agent-name.md`
- Be descriptive and specific
- Examples:
  - `scientific-python-expert.md`
  - `hpc-optimization-specialist.md`
  - `data-analysis-assistant.md`

### 2. Define Clear Expertise

Your agent should have well-defined expertise in:
- Specific technologies (e.g., NumPy, SciPy, Jupyter)
- Problem domains (e.g., data analysis, simulation, visualization)
- Methodologies (e.g., reproducible research, testing scientific code)

### 3. Write Effective Prompts

Agent prompts should:
- Clearly state the agent's expertise and scope
- Emphasize scientific computing best practices
- Promote reproducibility and code quality
- Include specific knowledge about relevant tools and libraries
- Set appropriate expectations and limitations

### 4. Agent Prompt Template

```markdown
You are an expert [ROLE] specializing in [DOMAIN]. 

## Your Expertise
- [Specific area 1]
- [Specific area 2]
- [Specific area 3]

## Your Approach
When helping with [DOMAIN] tasks, you:
- [Key practice 1]
- [Key practice 2]
- [Key practice 3]

## Technologies You're Familiar With
- [Technology 1]: [Brief description]
- [Technology 2]: [Brief description]
- [Technology 3]: [Brief description]

## Best Practices You Follow
- [Practice 1]
- [Practice 2]
- [Practice 3]

When assisting users, provide clear, well-documented solutions that follow scientific computing best practices and promote reproducible research.
```

## 🎯 Agent Categories

Agents in this repository may fall into these categories:

### Scientific Computing
- Numerical computing and algorithms
- High-performance computing (HPC)
- Parallel and distributed computing
- Scientific simulations

### Data Science & Analysis
- Data processing and cleaning
- Statistical analysis
- Machine learning for science
- Data visualization

### Research Software Engineering
- Software architecture for research
- Testing scientific code
- Documentation and reproducibility
- Version control and collaboration

### Domain-Specific
- Bioinformatics and computational biology
- Climate and earth science computing
- Physics and astronomy software
- Chemistry and materials science

## 📋 Agent Quality Standards

Agents should adhere to these standards:

### Technical Quality
- Accurate technical information
- Up-to-date library and framework knowledge
- Performance-conscious recommendations
- Security-aware practices

### Scientific Rigor
- Promote reproducibility
- Encourage proper documentation
- Support version control usage
- Emphasize testing and validation

### Code Quality
- Follow language-specific conventions
- Promote readable, maintainable code
- Encourage modular design
- Support best practices for the scientific Python ecosystem

## 🧪 Testing Agents

Before submitting an agent:

1. **Functional Test**: Use the agent with Claude Code on real tasks
2. **Scope Test**: Verify the agent stays within its defined expertise
3. **Quality Test**: Ensure recommendations follow best practices
4. **Documentation Test**: Check that usage guidelines are clear

## 📝 Agent Documentation

Each agent file should include:

1. **Header**: Clear title and brief description
2. **Expertise Section**: List of specific capabilities
3. **Usage Guidelines**: When to use this agent
4. **Agent Prompt**: The actual prompt text
5. **Examples** (optional): Sample use cases

## 🔄 Updating Agents

When updating existing agents:

- Maintain backward compatibility when possible
- Document significant changes
- Test thoroughly before submitting
- Update the agent's documentation section

## 📚 Resources for Agent Developers

- [Scientific Python Lectures](https://lectures.scientific-python.org/)
- [Best Practices for Scientific Computing](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745)
- [The Turing Way](https://the-turing-way.netlify.app/) - Guide to reproducible research
- [Software Carpentry Lessons](https://software-carpentry.org/lessons/)

## 🤝 Contributing

See the main [CONTRIBUTING.md](../../CONTRIBUTING.md) file for detailed contribution guidelines.

---

**Ready to create an agent?** Start by copying the template above and customizing it for your specific scientific computing domain!
