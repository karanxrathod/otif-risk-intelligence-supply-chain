# 🚚 OTIF Management – AI-Powered Supply Chain Control Tower

**Program:** UK–India Advanced AI Bootcamp (2-Day Program)
**Use Case:** OTIF Management (On-Time, In-Full Delivery)
**Theme:** Advanced AI Applications for Business Operations

---

## 📌 Project Overview

This project is an **AI-driven OTIF (On-Time, In-Full) Management System** designed as part of the **UK–India Advanced AI Bootcamp**. The objective is to demonstrate how **data, machine learning, and decision intelligence** can be applied to solve real-world supply chain challenges.

Modern supply chains operate across **distributed warehouses, dark stores, and multiple fulfillment partners**. OTIF failures rarely occur due to a single issue at the last mile; instead, they emerge from **inventory imbalances, demand uncertainty, delayed replenishment, and suboptimal order routing**.

This solution shifts OTIF management from **reactive firefighting** to **proactive, predictive planning** using AI.

---

## 🎯 Business Problem (Use Case 2: OTIF Management)

For FMCG, retail, and e-commerce businesses:

* Missed or partial deliveries reduce customer satisfaction
* Retail partners impose penalties for OTIF failures
* Operations teams lack early visibility into delivery risks

### Core Challenge

> *How can we predict OTIF failures in advance and enable timely operational interventions?*

---

## 💡 Solution Summary

The system continuously monitors:

* Inventory levels
* Incoming orders
* Demand patterns
* Fulfillment constraints

Using **machine learning**, it predicts **OTIF risk at the order level**, enabling operations teams to:

* Rebalance inventory
* Reroute orders
* Trigger replenishment actions

before a failure actually occurs.

---

## 🧩 Day 1 Deliverables (As per Bootcamp Guidelines)

### 1️⃣ Stakeholders & KPIs

**Key Stakeholders**

* Supply Chain Managers
* Warehouse & Fulfillment Teams
* Operations Planning Teams
* Business Leadership
* Data & AI Teams

**Key KPIs**

* OTIF % (On-Time, In-Full)
* On-Time Delivery Rate
* In-Full Delivery Rate
* Number of High-Risk Orders
* Potential OTIF Improvement from Interventions

---

### 2️⃣ Business Process Mapping

**Before AI**

* OTIF issues identified *after* delivery failures
* Manual investigation and reactive actions
* Limited visibility into future risks

**After AI**

* Predictive identification of OTIF risks
* Early alerts for high-risk orders
* Data-driven decision-making and simulations

---

### 3️⃣ CRISP-DM Mapping

| CRISP-DM Stage         | Implementation                             |
| ---------------------- | ------------------------------------------ |
| Business Understanding | OTIF improvement & cost reduction          |
| Data Understanding     | Orders, customers, products, delivery data |
| Data Preparation       | Feature engineering, cleaning, merging     |
| Modeling               | Random Forest classifier for OTIF risk     |
| Evaluation             | Accuracy, Recall, ROC-AUC                  |
| Deployment             | Streamlit-based Control Tower              |

---

## 🧠 AI & Data Science Approach

### Data Sources

* Customer dimension data
* Product & category data
* Order lines and aggregate order facts
* Delivery timelines and quantities

### Feature Engineering

* Planned lead time
* Demand rolling averages
* Demand volatility
* Order timing (day, month)
* Encoded categorical features (customer, category, city)

### Model

* **Algorithm:** Random Forest Classifier
* **Objective:** Predict probability of OTIF failure
* **Output:** Risk score per order

---

## 📊 System Capabilities

* 📈 Executive OTIF dashboard
* 🔮 Demand sensing and volatility tracking
* 🚨 High-risk order detection
* 🛠️ What-if simulation for proactive interventions
* 📥 Downloadable action list for operations teams

---

## 🖥️ Technology Stack

* Python
* Streamlit
* Pandas & NumPy
* Scikit-learn
* Plotly (interactive visualizations)

---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
streamlit run otif_dashboard.py
```

---

## 📈 Business Impact

* Early detection of OTIF risks
* Reduced delivery penalties
* Improved customer satisfaction
* Better inventory and fulfillment planning
* Clear shift from reactive to proactive operations

---

## 🚀 Bootcamp Outcome

This project demonstrates:

* Applied AI for real business problems
* End-to-end CRISP-DM thinking
* Strong alignment between business KPIs and ML outputs
* Practical deployment mindset via dashboards

---

## 👤 Author

**Karan Rathod**
Artificial Intelligence & Data Science Engineer
Project built as part of the **UK–India Advanced AI Bootcamp**

---

> *"From firefighting to foresight — AI-powered OTIF management."*
