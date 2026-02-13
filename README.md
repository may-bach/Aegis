# Aegis
> **Intelligence Shared. Privacy Shielded.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Federated Learning](https://img.shields.io/badge/Framework-Flower-orange)](https://flower.dev/)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()

## 🛡️ Overview

**Aegis** is a privacy-preserving machine learning initiative designed to solve the "data silo" problem in healthcare. Traditional AI requires centralizing data, which creates massive privacy risks and legal bottlenecks. Aegis bypasses this by bringing the model to the data, not the data to the model.

Using **Federated Learning (FL)**, Aegis simulates a network of hospitals (clients) that train a local model on their own private patient records. These clients send only mathematical model updates (gradients/weights)—never patient data—to a central coordinator, which aggregates them into a superior global model.

**Goal:** Accurately predict heart disease presence using the **UCI Heart Disease dataset** across distributed, non-IID data partitions while maintaining 100% data privacy.

## ⚡ Key Features

* **Decentralized Training:** Simulates 4–8 distinct hospital clients with unique patient demographics.
* **Privacy-First Architecture:** Raw data remains strictly on local client devices; only model parameters are transmitted.
* **Federated Averaging:** Implements the `FedAvg` strategy to aggregate insights from diverse sources.
* **Scalable Design:** Built with **Flower (flwr)** and **PyTorch**, designed to be compatible with cloud-native workflows (e.g., AWS SageMaker).

## 🏗️ Architecture

1.  **Global Server:** Initializes the model and distributes it to available clients.
2.  **Local Clients (Hospitals):**
    * Receive the global model.
    * Train locally on private data (UCI Heart Disease subset).
    * Calculate updates (weights) and send them back.
3.  **Aggregation:** The server averages the updates to create a new, smarter global model.
4.  **Repeat:** This cycle continues for varying rounds until accuracy converges.

## 🛠️ Tech Stack

* **Core Language:** Python 3.x
* **FL Framework:** Flower (`flwr`)
* **ML Backend:** PyTorch / Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Dataset:** UCI Heart Disease (Cleveland)

## 🗺️ Roadmap

- [ ] **Phase 1: Data Pipeline**
    - Load UCI Heart Disease dataset.
    - Implement non-IID data splitting (simulating uneven patient distributions across hospitals).
- [ ] **Phase 2: Client-Server Setup**
    - Define the `FlowerClient` class for local training.
    - Set up the server-side aggregation strategy (`FedAvg`).
- [ ] **Phase 3: Simulation**
    - Run multi-round simulation (10-50 rounds).
    - Benchmark Global Model Accuracy vs. Local-Only Models.
- [ ] **Phase 4: Optimization & Docs**
    - Visualize accuracy curves (Matplotlib/Seaborn).
    - Finalize documentation for AWS SageMaker compatibility.

## 🤝 Contributing

This project is currently in the initial development phase. Suggestions for privacy-preserving techniques (DP-SGD) or model architecture improvements are welcome.

---
*Built by [maybach uk]*
