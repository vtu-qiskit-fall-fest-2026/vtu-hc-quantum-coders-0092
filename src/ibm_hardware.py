"""IBM Quantum real hardware execution stub."""

import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv


def run_on_real_backend(
    circuit_or_model: Any,
    backend_name: Optional[str] = None,
    channel: str = "ibm_quantum",
) -> Dict[str, Any]:
    """Execute quantum circuits or evaluate model on real IBM Quantum hardware.

    Reads IBM_QUANTUM_API_KEY from environment/.env and verifies remaining
    allocation/usage quota before submitting jobs.

    Args:
        circuit_or_model: Quantum circuit, list of circuits, or quantum model to execute.
        backend_name: Name of target real backend (e.g. 'ibm_torino', 'ibm_brisbane').
        channel: IBM runtime channel (defaults to 'ibm_quantum').

    Returns:
        Job execution results, remaining quota information, and execution status.
    """
    load_dotenv()
    api_key = os.getenv("IBM_QUANTUM_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("IBM_QUANTUM_API_KEY is not set or still default in .env")

    # TODO: Initialize QiskitRuntimeService(channel=channel, token=api_key)
    # TODO: Check account usage / remaining quantum time quota before dispatching job
    # TODO: Submit job and return results
    raise NotImplementedError("TODO: Implement run_on_real_backend")
