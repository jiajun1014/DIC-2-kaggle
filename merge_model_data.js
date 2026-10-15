const fs = require('fs');

const startupsData = JSON.parse(fs.readFileSync('model_data.json', 'utf8'));
const tsmcData = JSON.parse(fs.readFileSync('tsmc_model_data.json', 'utf8'));

const problemsRegistry = {
  "50_startups": {
    id: "50_startups",
    name: "50 Startups Profit Prediction",
    subtitle: "Multiple Linear Regression & Feature Selection Explorer (CRISP-DM)",
    breadcrumb: "Regression Analysis > 50 Startups Feature Progression",
    heroTitle: "Feature Selection & Parsimony Analysis",
    heroDesc: "Compare 5 Machine Learning Feature Selection Algorithms (Lasso, Forward SFS, RFE, ElasticNet, Exhaustive Search) against Baseline OLS on the 50 Startups dataset. Explore the bias-variance tradeoff across feature counts k=1..5, pinpointing the 2-feature optimal sweet spot and the 5-feature overfitting cliff.",
    categoryBadge: "Kaggle Top 50 Startups",
    targetName: "Profit",
    targetUnit: "$",
    sweetSpotK: 2,
    overfittingK: 5,
    heroBadges: [
      { label: "50 Startups Dataset", icon: "fa-database" },
      { label: "5 Selection Methods", icon: "fa-sliders" },
      { label: "Global Sweet Spot: k=2 ($8,198.80 RMSE)", icon: "fa-award" },
      { label: "Overfitting Warning: k=5 ($9,137.99 RMSE)", icon: "fa-triangle-exclamation" }
    ],
    kLabels: {
      1: "k = 1 (R&D Spend)",
      2: "k = 2 ★ Optimal Sweet Spot (Min RMSE)",
      3: "k = 3 (+ State / Admin)",
      4: "k = 4 (4 Predictors)",
      5: "k = 5 ⚠️ Full Model (Overfitting Cliff)"
    },
    defaultInputs: {
      rd: 100000,
      mkt: 150000,
      admin: 120000,
      state: "California"
    },
    data: startupsData
  },
  "tsmc_2330": {
    id: "tsmc_2330",
    name: "TSMC (2330.TW) Stock Auto-Regression",
    subtitle: "Auto-Regression (AR) & Time-Lagged Feature Selection (CRISP-DM)",
    breadcrumb: "Financial AIoT > TSMC 2330.TW Auto-Regression (CRISP-DM)",
    heroTitle: "TSMC (2330.TW) Auto-Regression & Time-Lagged Feature Selection",
    heroDesc: "Evaluate Auto-Regressive (AR) Multiple Linear Regression on Taiwan Semiconductor Manufacturing Co. (2330.TW) daily closing prices. Test lagged price features (Lag 1, 2, 3, 5) and 5-Day Moving Average (MA 5) across 5 feature selection methods. Validate Occam's razor against high multicollinearity in financial time series.",
    categoryBadge: "Taiwan Semiconductor (2330.TW)",
    targetName: "Close Price",
    targetUnit: "NT$",
    sweetSpotK: 2,
    overfittingK: 5,
    heroBadges: [
      { label: "TSMC 2330.TW Dataset", icon: "fa-chart-line" },
      { label: "Auto-Regression AR(p) + MA", icon: "fa-clock" },
      { label: "Optimal Sweet Spot: k=2 (Lag 1 + MA 5)", icon: "fa-award" },
      { label: "Multicollinearity Warning: k=5 (VIF > 10)", icon: "fa-triangle-exclamation" }
    ],
    kLabels: {
      1: "k = 1 (Lag 1 Yesterday)",
      2: "k = 2 ★ Optimal Sweet Spot (Lag 1 + MA 5)",
      3: "k = 3 (+ Lag 2)",
      4: "k = 4 (4 Lag Features)",
      5: "k = 5 ⚠️ Full AR Model (High VIF Cliff)"
    },
    defaultInputs: {
      lag1: 820,
      ma5: 815,
      lag2: 810,
      lag3: 805,
      lag5: 800
    },
    data: tsmcData
  }
};

// Write out model_data.js
const jsContent = `// Auto-generated Comprehensive Model Data Registry for Multiple Problems
window.PROBLEMS_DATA = ${JSON.stringify(problemsRegistry, null, 2)};

// Default backward compatibility
window.MODEL_DATA = window.PROBLEMS_DATA["50_startups"].data;
`;

fs.writeFileSync('model_data.js', jsContent, 'utf8');
fs.writeFileSync('static/model_data.js', jsContent, 'utf8');

// Also update combined model_data.json
fs.writeFileSync('model_data.json', JSON.stringify(startupsData, null, 2), 'utf8');
fs.writeFileSync('all_problems_data.json', JSON.stringify(problemsRegistry, null, 2), 'utf8');

console.log('Successfully generated merged model_data.js and all_problems_data.json!');
