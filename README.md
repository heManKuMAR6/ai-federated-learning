# FederatedLearningCentralServer

## Description

A Python-based Federated Learning Central Server for aggregating AI model weights from distributed clients. This project implements a central server for federated learning, allowing multiple local models to contribute to a central AI model without sharing raw data. The central model aggregates weights, performs training, and improves through collaboration with decentralized local models.

## Features
- Federated learning setup
- Central server for model aggregation
- Local clients can submit model weights
- Aggregation of weights from multiple clients
- Error handling for mismatched model architectures
- Training and evaluation of the aggregated model

## Requirements

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
