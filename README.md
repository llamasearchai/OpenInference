# OpenInference

**[ALPHA - Under Development]**

OpenInference is a Python-based framework designed to streamline and optimize the deployment and serving of machine learning models, with a focus on Large Language Models (LLMs). It provides tools for model management, inference optimization, performance monitoring, and easy integration into various environments.

## Features

*   **Model Management**: Easily load, unload, and manage different versions of your ML models. Supports various model types, with a focus on transformer-based architectures.
*   **Optimized Runtimes**: Integrates with various hardware acceleration backends (CUDA, Metal planned) and optimization techniques (quantization, graph optimization planned) to deliver high-performance inference.
*   **Comprehensive Monitoring**: Built-in metrics collection for latency, throughput, resource utilization, and more. Export metrics for analysis and alerting.
*   **Batching Support**: Automatic request batching to improve throughput for concurrent requests.
*   **Extensible**: Designed with modularity in mind, allowing for easy extension and integration with custom components.
*   **CLI & API Access**: Interact with the inference server via a command-line interface or a programmatic API (REST API planned).

## Project Structure

```
OpenInference/
├── api/                  # (Planned) REST API for inference and management
├── batching/             # Request batching logic
├── docs/                 # Project documentation
├── hardware/             # Hardware-specific acceleration backends (CUDA, Metal)
├── integrations/         # Integrations with other tools and platforms
├── memory/               # Memory management utilities (e.g., KV cache)
├── models/
│   ├── converters/       # Model format conversion tools
│   └── loaders/          # Model loading utilities
├── monitoring/           # Metrics collection and monitoring
├── optimization/
│   ├── graph/            # (Planned) Graph optimization techniques
│   ├── kernels/          # (Planned) Custom optimized kernels
│   └── quantization/     # (Planned) Model quantization tools
├── runtime/
│   ├── core/             # Core inference runtime logic
│   ├── cuda/             # CUDA-specific runtime components
│   └── metal/            # (Planned) Metal-specific runtime components
├── tests/                # Unit and integration tests
├── ui/                   # (Planned) Web UI for monitoring and management
├── __init__.py
├── .env                  # Environment configuration (template as .env.example)
├── cli.py                # Command-line interface
├── main.py               # Main application entry point and OpenInference server
├── setup.py              # Packaging and installation script
└── LICENSE               # Project License
```

## Getting Started

### Prerequisites

*   Python 3.8+
*   Pip (Python package installer)
*   (Optional but Recommended) NVIDIA GPU with CUDA drivers for GPU acceleration.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/llamasearchai/OpenInference.git
    cd OpenInference
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -e . # Install OpenInference in editable mode
    ```
    *Note: `requirements.txt` provides a list of core dependencies. For development, also install `requirements-dev.txt` as shown below.*

    For development, including tools for testing, linting, and documentation, install:
    ```bash
    pip install -r requirements-dev.txt
    ```

4.  **(Optional) Create a `.env` file for environment-specific configurations:**
    You can copy `.env.example` (if it exists, otherwise create one) to `.env` and customize it.
    Example `.env` content:
    ```env
    LOG_LEVEL=INFO
    METRICS_EXPORT_PATH=/tmp/openinference_metrics
    # HUGGING_FACE_HUB_TOKEN=your_hf_token_here # If needed for private models
    ```

### Running Tests

To ensure everything is set up correctly and the core functionalities are working, run the tests:

```bash
python -m unittest discover -s tests
```

## Usage

OpenInference provides a Command Line Interface (`cli.py`) for managing and interacting with the inference server.

### Starting the Inference Server

(This functionality is primarily within `main.py` and might be expanded via `cli.py` in the future)

```bash
popeninference --help # To see available commands
```

### CLI Examples

*   **List available models (locally or from a registry):**
    ```bash
    popeninference list-models
    ```

*   **Load a model:**
    ```bash
    popeninference load-model --model-id "TheBloke/Mistral-7B-Instruct-v0.1-GGUF" --alias "mistral-7b-instruct"
    ```
    *(Ensure you have a GGUF compatible model or adjust the model ID accordingly)*

*   **Run inference (example for a loaded text generation model):**
    ```bash
    popeninference run-inference --model-alias "mistral-7b-instruct" --prompt "What is the capital of France?"
    ```

*   **Unload a model:**
    ```bash
    popeninference unload-model --model-alias "mistral-7b-instruct"
    ```

### Monitoring

Metrics are collected automatically and can be exported to a specified path (see `METRICS_EXPORT_PATH` in `.env`).
Future versions will include a web UI for real-time monitoring.

## Contributing

We welcome contributions to OpenInference! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute, report issues, and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

*   (TODO: Add any acknowledgements if necessary, e.g., libraries, research papers that inspired the project)

---

## Roadmap / Future Work

OpenInference is actively under development. Here are some of the features and improvements we are planning:

*   **Full REST API**: Comprehensive API for model management, inference, and monitoring.
*   **Web UI**: A user-friendly web interface for monitoring server status, model performance, and managing models.
*   **Advanced Optimization Techniques**: 
    *   Deeper integration with TensorRT for NVIDIA GPUs.
    *   Model quantization (AWQ, GPTQ, etc.).
    *   Graph optimization and kernel fusion.
*   **Hardware Acceleration Expansion**:
    *   Enhanced CUDA support with custom kernels.
    *   Support for Apple Metal for macOS devices.
    *   (Potentially) Support for other hardware accelerators (e.g., TPUs, AMD GPUs).
*   **Expanded Model Support**: Wider compatibility with various model architectures and formats.
*   **Distributed Inference**: Support for serving very large models across multiple GPUs or nodes.
*   **Python Package (PyPI) Release**: Formal packaging and release on PyPI.
*   **Comprehensive Documentation**: Detailed documentation for users and developers, including API references and tutorials.
*   **Example Integrations**: Showcase integrations with popular MLOps tools and platforms.

Stay tuned for updates!

*This README is a work in progress and will be updated as the project evolves.* 