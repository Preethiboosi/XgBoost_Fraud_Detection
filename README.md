# XgBoost_Fraud_Detection
a REST API and an end-to-end machine learning pipeline for detecting fraudulent financial transactions. This project uses XGBoost and FastAPI to construct a modular, production-ready architecture, going beyond basic "tutorial" scripts.

Crucial "Upgraded" Elements:
1.Built with a distinct division of responsibilities, the modular architecture includes specific modules for real-time inference, feature engineering, and model training.
2.Advanced Imbalance Handling: To ensure that the model gives priority to detecting infrequent fraudulent events, an extraordinary 171:1 class imbalance (Legit vs. Fraud) was handled using XGBoost's scale_pos_weight option.
3.Big Data Engineering: Over 1.2 million transaction records (about 350 MB of data) were successfully processed and trained.
4.Real-Time Serving: An interactive Swagger user interface was used to deploy the model using a FastAPI REST server, enabling immediate transaction validation.
Source: Kaggle - Credit Card Transactions Fraud Detection Dataset
