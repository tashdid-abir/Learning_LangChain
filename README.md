# Learning LangChain

A hands-on repository documenting my journey of learning and experimenting with LangChain through practical Python examples.

The code in this repository will grow over time as I explore new ideas, test different approaches, and build a stronger understanding of working with language models and AI applications.

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd Learning_LangChain
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add the credentials and configuration values required by the examples you want to run.

For examples using OpenRouter, the file may contain:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=your_model_name_here
```

Do not commit real API keys or other secrets to the repository.

## Project Structure

The repository is organized into numbered folders. Each folder contains small, focused examples that build on the concepts introduced earlier. New sections and experiments will be added as the learning journey continues.

## Running an Example

Activate the virtual environment and run an example with Python:

```powershell
python path\to\example.py
```

For example:

```powershell
python 5_Chains\4_conditional_chain.py
```

The exact requirements may vary between examples, so check the source file and environment configuration before running it.

## Purpose

This repository is intended for learning, experimentation, and reference. The examples may change as concepts are revisited and improved.
