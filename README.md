# 🌐 AIoT CRISP-DM 多元線性回歸與特徵選取互動分析平台

本專案遵循 **CRISP-DM**（Cross-Industry Standard Process for Data Mining，跨產業資料探勘標準流程）方法論，建構一套結合機器學習、統計推論與全端互動視覺化的多元線性回歸（Multiple Linear Regression, OLS）分析平台。

平台整合了 **5 大特徵選取演算法**（Lasso 正則化、Forward SFS 順向特徵選取、RFE 遞迴特徵消除、ElasticNet 彈性網路、Exhaustive Search 最佳子集窮舉搜尋），探討特徵數量 $k=1 \dots 5$ 下的偏誤與變異權衡（Bias-Variance Tradeoff），並支援**多題目即時切換**以及**任意資料集瀏覽器端即時建模**（加分亮點）。

---

## 🌐 線上展示與互動網站 (Live Demo)

* 🐙 **GitHub 專案儲存庫**：**[https://github.com/jiajun1014/DIC-2-kaggle](https://github.com/jiajun1014/DIC-2-kaggle)**  
* 🔗 **GitHub Pages 線上展示**：**[https://jiajun1014.github.io/DIC-2-kaggle/](https://jiajun1014.github.io/DIC-2-kaggle/)**

![AIoT 多元線性回歸互動儀表板](dashboard_demo.png)

> 🎨 **視覺風格設計**：參考專業 SaaS 統計計算器介面風格 [StatsUnlock](https://statsunlock.com/multiple-linear-regression-calculator/)，結合現代科技暗色漸層、動態卡片與響應式資料視覺化佈局。

---

## 🌟 作業加分亮點功能完整實現

### 1. 📌 頂部下拉選單即時呈現多題目切換（加總分 1 分）
儀表板頂端設有直覺的 **選擇題目 (Problem Selector)** 下拉選單，可無縫切換不同商業與工程領域的預測主題，並動態重新渲染特徵演算法、回歸模型與互動試算器：

* **Problem 1：50 Startups 新創公司獲利預測（Kaggle 經典多元線性回歸）**
  * **目標變數 ($y$)**：`Profit`（新創公司淨利潤，美元 $）
  * **候選特徵 ($X$)**：`R&D Spend`（研發支出）、`Administration`（行政支出）、`Marketing Spend`（行銷支出）、`State`（加州 / 佛州 / 紐約州，經 One-Hot Encoding）
  * **核心統計發現**：驗證 $k=2$（`R&D Spend` + `Marketing Spend`）為最佳精簡模型（符合奧坎剃刀原則 Occam's Razor），測試集 RMSE 達到全域最低 $8,198.80；當納入全部 5 項特徵時，模型陷入過擬合峭壁（Overfitting Cliff），測試誤差暴增至 $9,137.99。

* **Problem 2：台積電 2330.TW 股價自回歸預測（時間序列 Auto-Regression）**
  * **目標變數 ($y$)**：`Close Price`（當日收盤價，新台幣 NT$）
  * **候選特徵 ($X$)**：時間滯後特徵（`Lag 1 昨日收盤價`、`Lag 2 前日收盤價`、`Lag 3`、`Lag 5`、`MA 5 五日移動平均線`）
  * **核心統計發現**：驗證時間序列中 $k=2$（`Lag 1` + `MA 5`）具備最佳泛化能力（Test RMSE 僅 4.68 元、$R^2 = 0.9852$）；當特徵數擴展至 $k=5$ 時，相鄰交易日極高的自相關性導致嚴重多元共線性（VIF > 10），統計檢定顯著度顯著下降。

---

### 2. 🌐 任意資料集即時載入分析平台（加總分 2 分）
平台內建完整的 **Arbitrary Dataset Live CRISP-DM Studio**，具備全自動瀏覽器端機器學習引擎，支援多種資料來源：

* **自訂公開網址輸入**：可輸入任意 CSV 檔案 URL（如 GitHub Raw CSV 連結）。
* **內建一鍵範例資料集**：
  * 🏢 50 Startups（Avik Jain Kaggle 經典資料集）
  * 📈 台積電 2330.TW 歷史股價自回歸資料集 (`tsmc_2330_stock.csv`)
  * 🏠 Boston Housing 波士頓房價回歸資料集 (`BostonHousing.csv`)
* **本機 CSV 檔案上傳**：可直接拖放或選取電腦內的 `.csv` 檔案，完全免除瀏覽器跨域（CORS）限制。
* **貼上 CSV 文字**：提供快速貼上文字對話框，即時貼入試算表資料。
* **瀏覽器端全自動 CRISP-DM 流程處理**：
  1. **商業與資料理解 (Data Understanding)**：自動判斷欄位資料型態、有效樣本數、各特徵描述性統計（平均數、標準差、最大最小值）。
  2. **資料準備 (Data Preparation)**：支援使用者動態勾選預測目標變數 $y$ 與自變數 $X$；類別型變數自動執行虛擬變數編碼（One-Hot Encoding），並自動剔除基準組以徹底避免**虛擬變數陷阱 (Dummy Variable Trap)**；自動執行 80/20 訓練/測試集分割。
  3. **模型建立 (Modeling)**：利用矩陣正規方程 $\beta = (X^T X)^{-1} X^T y$ 在 JavaScript 內進行即時 OLS 精確求解。
  4. **特徵選取與評估 (Feature Selection & Evaluation)**：即時執行順向逐步特徵選取（Forward Stepwise Selection），動態輸出各階模型的 RMSE、$R^2$、調整後 $R^2$ (Adjusted $R^2$)、MAE、F 檢定統計量、標準誤、t 檢定值、p-value 及 VIF 共線性指標。
  5. **模型部署與應用 (Deployment)**：針對最佳模型自動生成對應的特徵數值輸入滑桿，支援即時推論預測與測試集殘差分佈診斷表。

---

## 📊 五大特徵選取演算法實證成效比較

本專案使用 Python `scikit-learn` 與 `statsmodels` 針對 50 Startups 資料集進行完整特徵演算法驗證：

| 特徵選取演算法 | $k=1$ 最佳特徵 | $k=2$ 甜蜜點特徵組合 (Sweet Spot) | 甜蜜點 Test RMSE | Test $R^2$ | 統計檢定 $F$ 統計量 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Lasso (L1 正則化)** | R&D Spend | **R&D Spend + Marketing Spend** | **$8,198.80** | **0.9474** | **349.0 ($p < 0.001$)** |
| **Forward SFS (順向選取)** | R&D Spend | **R&D Spend + Marketing Spend** | **$8,198.80** | **0.9474** | **349.0 ($p < 0.001$)** |
| **RFE (遞迴消除)** | R&D Spend | **R&D Spend + Marketing Spend** | **$8,198.80** | **0.9474** | **349.0 ($p < 0.001$)** |
| **ElasticNet (彈性網路)** | R&D Spend | **R&D Spend + Marketing Spend** | **$8,198.80** | **0.9474** | **349.0 ($p < 0.001$)** |
| **Exhaustive (窮舉搜尋)** | R&D Spend | **R&D Spend + Marketing Spend** | **$8,198.80** | **0.9474** | **349.0 ($p < 0.001$)** |

### 核心統計推論結論：
1. **五大演算法一致收斂**：在 $k=2$ 時，所有 5 種演算法一致挑選出 `['R&D Spend', 'Marketing Spend']` 作為最佳特徵組合。
2. **奧坎剃刀原則 (Occam's Razor)**：模型僅需 2 個特徵即可解釋 94.74% 的目標變異，其餘特徵（`Administration`、`State`）的 t 檢定 p-value 均大於 0.05，未達統計顯著水準。
3. **過擬合峭壁效應 (Overfitting Cliff)**：當特徵數量自 $k=2$ 增加至 $k=5$ 時，訓練集誤差雖略有降低，但測試集 RMSE 卻由 $8,198.80 反彈暴增至 $9,137.99，充分印證模型過度配適之現象。

---

## 💻 本機安裝與執行方式 (Local Setup & Run)

### 方式一：Node.js 輕量本地伺服器（推薦）
本專案內建免安裝額外依賴的 Node.js 伺服器，直接執行：
```bash
node dev_server.js
```
於瀏覽器開啟：**[http://localhost:3000/](http://localhost:3000/)**

### 方式二：Python 簡易 HTTP 伺服器
```bash
python -m http.server 8000
```
於瀏覽器開啟：**[http://localhost:8000/](http://localhost:8000/)**

### 方式三：Flask 後端 REST API 應用
```bash
pip install -r requirements.txt
python app.py
```
於瀏覽器開啟：**[http://localhost:5000/](http://localhost:5000/)**

#### 支援的 Flask REST API 端點：
* `GET /api/problems`：取得目前註冊的分析題目清單與規格
* `GET /api/models?problem=50_startups&algo=Lasso`：取得特定題目與演算法之各特徵階層評估數據
* `POST /api/predict`：即時推論運算端點，支援自訂數值傳入並回傳特徵貢獻拆解

---

## 📁 專案檔案架構

```text
├── index.html                          # 前端主要儀表板（含題目切換與任意資料集分析 Studio）
├── model_data.js                       # 跨問題資料集與各階回歸模型結果登錄檔
├── tsmc_2330_stock.csv                 # 台積電歷史自回歸特徵資料集
├── dev_server.js                       # 輕量本機 Node.js 靜態 HTTP 伺服器
├── app.py                              # Flask 後端伺服器與 REST API
├── templates/
│   └── index.html                      # Flask 後端專用模板
├── static/
│   └── model_data.js                   # 靜態資源同步檔
├── main.py                             # Python CRISP-DM 終端建模與評估腳本
├── feature_progression_analysis.py     # 5 大特徵選取演算法分析腳本
├── generate_tsmc_model_data.py         # 台積電股價自回歸資料生成腳本
├── all_5_algorithms_comparison_plot.png # 5 大演算法特徵遞增對比圖表
├── dashboard_demo.png                  # 儀表板完整操作截圖
├── .github/workflows/pages.yml         # GitHub Pages 自動化部屬流程
└── README.md                           # 專案完整說明手冊
```

---

## 🙏 參考資料與致謝

1. **視覺介面與統計佈局參考**：[StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)
2. **50 Startups 資料集來源**：[Avik Jain 100-Days-Of-ML-Code](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)
3. **分析方法論標準**：CRISP-DM (Cross-Industry Standard Process for Data Mining)
