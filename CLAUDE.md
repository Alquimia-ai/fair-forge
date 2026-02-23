# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alquimia AI Fair Forge is a performance-measurement library for evaluating AI models and assistants. It provides metrics for fairness, toxicity, bias, conversational quality, and more.

## Development Commands

```bash
# Install dependencies
uv sync

# Run scripts in development
uv run python your_script.py

# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/metrics/test_toxicity.py

# Run a specific test
uv run pytest tests/metrics/test_toxicity.py::test_function_name

# Run tests in parallel
uv run pytest -n auto

# Skip slow tests
uv run pytest -m "not slow"

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy fair_forge

# Build package
uv build
```

## Design Principles

This project demands clean, well-architected object-oriented code. Every piece of code must reflect intentional design decisions. Think in terms of **SOLID principles** and **design patterns** before writing a single line.

### SOLID Principles — Non-Negotiable

These are not suggestions. Every class and function must comply:

- **S — Single Responsibility Principle**: Each class has exactly one reason to change. If a class handles both data loading and data transformation, split it. No exceptions.
- **O — Open/Closed Principle**: Code must be open for extension but closed for modification. Adding a new metric, guardian, storage backend, or strategy must NEVER require changing existing classes. You extend the system by adding new classes that implement existing interfaces.
- **L — Liskov Substitution Principle**: Any subclass must be usable wherever its parent class is expected without breaking behavior. If `LlamaGuard` replaces `IBMGranite`, the system must work identically through the `Guardian` interface.
- **I — Interface Segregation Principle**: Do not force classes to depend on methods they do not use. Prefer small, focused interfaces (Protocols or ABCs) over large monolithic ones.
- **D — Dependency Inversion Principle**: High-level modules must not depend on low-level modules. Both must depend on abstractions. Concrete classes are always injected, never instantiated directly inside business logic.

### Design Patterns — Mandatory Approach

Reference: https://refactoring.guru/design-patterns

- **No string-driven branching**: NEVER pass strings as parameters to select behavior. If you find yourself writing `if mode == "x"` / `elif mode == "y"`, stop immediately. Define a common interface (ABC or Protocol) and pass a class that implements it. Use **Strategy**, **Command**, or **Template Method** instead.
- **No long if/elif/else chains**: These are always a design failure. Replace them with polymorphism, a registry of classes, or the Strategy pattern.
- **Favor composition over inheritance**: Use **Decorator**, **Adapter**, and **Strategy** to compose behavior. Deep inheritance hierarchies are forbidden.
- **Program to interfaces, not implementations**: Depend on abstract base classes or Protocols. Concrete classes are injected from the outside.

### Patterns Already Used in This Project

Follow these established patterns when extending the codebase:

- **Template Method**: `FairForge` base class defines the `_process()` flow, subclasses implement `batch()`.
- **Strategy**: `FrequentistMode` / `BayesianMode` for statistical analysis, `SelectionStrategy` for generators.
- **Adapter**: Guardian implementations (`IBMGranite`, `LlamaGuard`) adapt different models behind a common `Guardian` interface.
- **Factory Method**: `MyMetric.run(RetrieverClass)` instantiates and orchestrates the pipeline.

When in doubt, ask: *"Can I solve this with a class and an interface instead of a conditional?"* — the answer is almost always yes.

## Code Style

- **Minimal comments**: Do not add inline comments explaining what the code does. The code should be self-explanatory through clear naming. Only comment *why* something is done when the reasoning is non-obvious.
- **No redundant docstrings**: Do not add docstrings to private methods, simple functions, or anything where the signature already conveys the intent.
- **No commented-out code**: Never leave commented-out code blocks. Either use the code or remove it.
- **No section dividers**: Do not add decorative comment blocks like `# --- Section ---` or `# ============`.

## Code Smells — Zero Tolerance

Reference: https://refactoring.guru/refactoring/smells

Every code smell listed below is a defect. Do not introduce any of them. If you encounter one while working on nearby code, refactor it.

### Bloaters

- **Long Method**: Break methods that grow beyond a single clear responsibility. Extract into smaller, well-named methods.
- **Large Class**: Split classes that accumulate too many fields or methods. Each class has one job.
- **Primitive Obsession**: Do not pass raw strings, ints, or dicts when a dedicated class or dataclass is appropriate. Wrap related primitives into value objects.
- **Long Parameter List**: If a method takes more than 3-4 parameters, introduce a parameter object or rethink the design.
- **Data Clumps**: Groups of variables that always appear together must be extracted into their own class.

### Object-Orientation Abusers

- **Switch Statements**: Replace `if/elif/else` and `match/case` chains that select behavior with polymorphism. Use Strategy or a class registry.
- **Temporary Field**: Fields that are only set in certain scenarios indicate a class doing too much. Extract into a separate class.
- **Refused Bequest**: If a subclass ignores or overrides most of its parent, the inheritance is wrong. Use composition instead.
- **Alternative Classes with Different Interfaces**: Classes that do the same thing must share a common interface. Unify them behind an ABC or Protocol.

### Change Preventers

- **Divergent Change**: If one class must be modified for multiple unrelated reasons, it violates SRP. Split it.
- **Shotgun Surgery**: If a single change requires touching many classes, the responsibility is scattered. Consolidate it.
- **Parallel Inheritance Hierarchies**: If adding a subclass in one hierarchy forces adding one in another, merge or decouple them.

### Dispensables

- **Duplicate Code**: Identical or near-identical logic in multiple places must be extracted into a shared method or base class.
- **Dead Code**: Unused classes, methods, variables, or imports must be deleted immediately. No keeping code "just in case".
- **Lazy Class**: If a class does not justify its existence, inline it or merge it.
- **Speculative Generality**: Do not create abstractions, parameters, or classes for hypothetical future use. Build what is needed now.
- **Excessive Comments**: Comments that explain *what* code does are a smell — the code itself should be clear. Only comment *why*.
- **Data Class**: Classes that only hold data with no behavior are suspect. Move behavior that operates on that data into the class itself.

### Couplers

- **Feature Envy**: A method that uses more from another class than its own belongs in that other class. Move it.
- **Inappropriate Intimacy**: Classes must not reach into the internals of other classes. Interact through public interfaces only.
- **Message Chains**: Long chains like `a.get_b().get_c().get_d()` expose internal structure. Use delegation or facade methods.
- **Middle Man**: If a class exists only to delegate to another class, remove it. It adds indirection with no value.
- **Incomplete Library Class**: When a library class lacks needed functionality, extend it via Decorator or Adapter — do not work around it with scattered conditionals.

## Architecture

### Core Pattern: FairForge Base Class
All metrics inherit from `FairForge` (in `fair_forge/core/base.py`). The pattern:
1. Subclass `FairForge`
2. Implement `batch()` method to process conversation batches
3. Append results to `self.metrics`
4. Use via `MyMetric.run(RetrieverClass, **kwargs)` which handles instantiation and processing

### Data Flow
```
Retriever.load_dataset() -> list[Dataset]
    ↓
FairForge._process() iterates datasets
    ↓
Metric.batch() processes each conversation
    ↓
Results in self.metrics
```

### Key Modules

- **`fair_forge/metrics/`**: Metric implementations (Toxicity, Bias, Context, Conversational, Humanity, BestOf, Agentic). Uses lazy imports.
- **`fair_forge/core/`**: Base classes - `FairForge`, `Retriever`, `Guardian`, `ToxicityLoader`, `SentimentAnalyzer`
- **`fair_forge/schemas/`**: Pydantic models for data validation (`Dataset`, `Batch`, metric-specific schemas)
- **`fair_forge/runners/`**: Test execution against AI systems (`BaseRunner`, `AlquimiaRunner`)
- **`fair_forge/storage/`**: Storage backends for test datasets (`LocalStorage`, `LakeFSStorage`)
- **`fair_forge/statistical/`**: Statistical modes (`FrequentistMode`, `BayesianMode`) for metrics like Toxicity
- **`fair_forge/guardians/`**: Bias detection implementations (IBMGranite, LlamaGuard)
- **`fair_forge/loaders/`**: Dataset loaders (e.g., `HurtlexLoader` for toxicity lexicons)
- **`fair_forge/llm/`**: LLM integration (`Judge`, prompts, schemas for structured outputs)
- **`fair_forge/extractors/`**: Group extraction implementations (`EmbeddingGroupExtractor`)
- **`fair_forge/utils/`**: Utilities (logging configuration)

### Custom Retriever Pattern
Users must implement a `Retriever` subclass to load their data:
```python
from fair_forge.core.retriever import Retriever
from fair_forge.schemas.common import Dataset

class MyRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        # Load and return datasets
        pass
```

### Test Fixtures
Shared fixtures in `tests/conftest.py` provide mock retrievers and datasets for each metric type (e.g., `toxicity_dataset_retriever`, `bias_dataset_retriever`).

## Key Data Structures

- **`Dataset`**: A conversation session with `session_id`, `assistant_id`, `language`, `context`, and `conversation` (list of Batch)
- **`Batch`**: Single Q&A interaction with `query`, `assistant`, `ground_truth_assistant`, `qa_id`
