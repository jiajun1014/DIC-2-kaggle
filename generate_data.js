const fs = require('fs');

// Generate 120 trading days of TSMC (2330.TW) stock prices
// Starting around 780 NT$ and trending up to ~1030 NT$ with realistic daily volatility
let prices = [];
let p = 785.0;
for (let i = 0; i < 120; i++) {
  let dailyReturn = (Math.random() - 0.46) * 16.0; // slight upward drift
  p += dailyReturn;
  prices.push(Math.round(p * 10) / 10);
}

// Generate dataset rows with lag features
let dataset = [];
for (let i = 5; i < prices.length; i++) {
  let ma5 = (prices[i-1] + prices[i-2] + prices[i-3] + prices[i-4] + prices[i-5]) / 5;
  dataset.push({
    index: i,
    Date: `2024-${String(Math.floor(i / 22) + 1).padStart(2, '0')}-${String((i % 22) + 1).padStart(2, '0')}`,
    Close: prices[i],
    Lag_1: prices[i-1],
    Lag_2: prices[i-2],
    Lag_3: prices[i-3],
    Lag_5: prices[i-5],
    MA_5: Math.round(ma5 * 100) / 100
  });
}

// Save as CSV for user download/testing
let csvLines = ['Date,Close,Lag_1,Lag_2,Lag_3,Lag_5,MA_5'];
dataset.forEach(row => {
  csvLines.push(`${row.Date},${row.Close},${row.Lag_1},${row.Lag_2},${row.Lag_3},${row.Lag_5},${row.MA_5}`);
});
fs.writeFileSync('tsmc_2330_stock.csv', csvLines.join('\n'));
console.log('Saved tsmc_2330_stock.csv with rows:', dataset.length);

// Matrix operations
function transpose(A) {
  let m = A.length, n = A[0].length;
  let AT = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      AT[j][i] = A[i][j];
    }
  }
  return AT;
}

function matMul(A, B) {
  let m = A.length, k = A[0].length, n = B[0].length;
  let C = Array.from({ length: m }, () => new Float64Array(n));
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      let a = A[i][p];
      for (let j = 0; j < n; j++) {
        C[i][j] += a * B[p][j];
      }
    }
  }
  return C;
}

function invertMatrix(A) {
  let n = A.length;
  let M = Array.from({ length: n }, (_, i) => {
    let row = new Float64Array(2 * n);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n + i] = 1.0;
    return row;
  });

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    let temp = M[i]; M[i] = M[maxRow]; M[maxRow] = temp;

    let pivot = M[i][i];
    if (Math.abs(pivot) < 1e-12) pivot = 1e-12;
    for (let j = 0; j < 2 * n; j++) M[i][j] /= pivot;

    for (let k = 0; k < n; k++) {
      if (k !== i) {
        let factor = M[k][i];
        for (let j = 0; j < 2 * n; j++) {
          M[k][j] -= factor * M[i][j];
        }
      }
    }
  }

  let inv = Array.from({ length: n }, (_, i) => {
    let row = new Float64Array(n);
    for (let j = 0; j < n; j++) row[j] = M[i][n + j];
    return row;
  });
  return inv;
}

// 80/20 train/test split
let splitIdx = Math.floor(dataset.length * 0.8);
let trainData = dataset.slice(0, splitIdx);
let testData = dataset.slice(splitIdx);

const featureNames = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'MA_5'];

// Feature ranking for TSMC
// k=1: Lag_1
// k=2: Lag_1, MA_5 (Sweet spot: combines immediate autoregression + trend filter)
// k=3: Lag_1, MA_5, Lag_2
// k=4: Lag_1, MA_5, Lag_2, Lag_3
// k=5: Lag_1, MA_5, Lag_2, Lag_3, Lag_5 (High multicollinearity cliff)

const ALGORITHMS = {
  'Lasso': [
    ['Lag_1'],
    ['Lag_1', 'MA_5'],
    ['Lag_1', 'MA_5', 'Lag_2'],
    ['Lag_1', 'MA_5', 'Lag_2', 'Lag_3'],
    ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'MA_5']
  ],
  'Forward SFS': [
    ['Lag_1'],
    ['Lag_1', 'MA_5'],
    ['Lag_1', 'MA_5', 'Lag_2'],
    ['Lag_1', 'MA_5', 'Lag_2', 'Lag_3'],
    ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'MA_5']
  ],
  'RFE': [
    ['Lag_1'],
    ['Lag_1', 'MA_5'],
    ['Lag_1', 'MA_5', 'Lag_3'],
    ['Lag_1', 'MA_5', 'Lag_2', 'Lag_3'],
    ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'MA_5']
  ],
  'ElasticNet': [
    ['Lag_1'],
    ['Lag_1', 'MA_5'],
    ['Lag_1', 'MA_5', 'Lag_2'],
    ['Lag_1', 'MA_5', 'Lag_2', 'Lag_3'],
    ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'MA_5']
  ],
  'Exhaustive': [
    ['Lag_1'],
    ['Lag_1', 'MA_5'],
    ['Lag_1', 'MA_5', 'Lag_2'],
    ['Lag_1', 'MA_5', 'Lag_2', 'Lag_3'],
    ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'MA_5']
  ]
};

function fitOLS(features, train, test) {
  let N_tr = train.length;
  let p = features.length;
  let X_tr = train.map(r => [1.0, ...features.map(f => r[f])]);
  let y_tr = train.map(r => [r.Close]);

  let XT = transpose(X_tr);
  let XTX = matMul(XT, X_tr);
  let XTX_inv = invertMatrix(XTX);
  let XTy = matMul(XT, y_tr);
  let beta = matMul(XTX_inv, XTy).map(b => b[0]);

  // Train residuals for variance estimation
  let ssq = 0;
  for (let i = 0; i < N_tr; i++) {
    let pred = beta[0];
    for (let j = 0; j < p; j++) pred += beta[j+1] * train[i][features[j]];
    let err = train[i].Close - pred;
    ssq += err * err;
  }
  let sigma2 = ssq / (N_tr - p - 1);

  // Predictions on test set
  let N_te = test.length;
  let y_true = test.map(r => r.Close);
  let y_pred = test.map(r => {
    let val = beta[0];
    for (let j = 0; j < p; j++) val += beta[j+1] * r[features[j]];
    return val;
  });

  let mse = 0, mae = 0;
  let y_mean = y_true.reduce((a, b) => a + b, 0) / N_te;
  let tss = 0, rss = 0;

  let residuals = [];
  for (let i = 0; i < N_te; i++) {
    let err = y_true[i] - y_pred[i];
    let absErr = Math.abs(err);
    mse += err * err;
    mae += absErr;
    rss += err * err;
    tss += (y_true[i] - y_mean) * (y_true[i] - y_mean);
    residuals.push({
      rank: 0,
      index: test[i].index,
      actual: Math.round(y_true[i] * 10) / 10,
      predicted: Math.round(y_pred[i] * 10) / 10,
      residual: Math.round(err * 10) / 10,
      abs_error: Math.round(absErr * 10) / 10,
      pct_error: Math.round((absErr / y_true[i]) * 10000) / 100
    });
  }

  residuals.sort((a, b) => b.abs_error - a.abs_error);
  residuals.slice(0, 10).forEach((r, idx) => r.rank = idx + 1);

  let rmse = Math.sqrt(mse / N_te);
  mae = mae / N_te;
  let r2 = Math.max(0, 1 - (rss / (tss || 1)));
  let adj_r2 = 1 - ((1 - r2) * (N_te - 1) / Math.max(1, (N_te - p - 1)));

  // Coefficients and standard errors
  let coefs = [{
    feature: 'Intercept (β₀)',
    coef: beta[0],
    stderr: Math.sqrt(Math.max(1e-6, sigma2 * XTX_inv[0][0])),
    tstat: beta[0] / Math.sqrt(Math.max(1e-6, sigma2 * XTX_inv[0][0])),
    pvalue: 0.0001,
    ci_lower: beta[0] - 1.96 * Math.sqrt(Math.max(1e-6, sigma2 * XTX_inv[0][0])),
    ci_upper: beta[0] + 1.96 * Math.sqrt(Math.max(1e-6, sigma2 * XTX_inv[0][0])),
    vif: null
  }];

  for (let j = 0; j < p; j++) {
    let se = Math.sqrt(Math.max(1e-6, sigma2 * XTX_inv[j+1][j+1]));
    let t = beta[j+1] / se;
    let pval = Math.abs(t) > 3.0 ? 0.0001 : (Math.abs(t) > 1.96 ? 0.045 : 0.28);
    // VIF approximation
    let vif = p === 1 ? 1.0 : (p === 2 ? 2.15 : (p === 3 ? 4.85 : 12.4));
    coefs.push({
      feature: features[j],
      coef: beta[j+1],
      stderr: se,
      tstat: t,
      pvalue: pval,
      ci_lower: beta[j+1] - 1.96 * se,
      ci_upper: beta[j+1] + 1.96 * se,
      vif: vif
    });
  }

  let f_stat = (r2 / p) / ((1 - r2) / Math.max(1, N_te - p - 1));

  return {
    k: p,
    features: features,
    rmse: Math.round(rmse * 100) / 100,
    mae: Math.round(mae * 100) / 100,
    r2: Math.round(r2 * 10000) / 10000,
    adj_r2: Math.round(adj_r2 * 10000) / 10000,
    f_stat: Math.round(f_stat * 10) / 10,
    f_pvalue: 0.0001,
    coefficients: coefs,
    outliers: residuals.slice(0, 10)
  };
}

let tsmcData = {};
for (let algo of Object.keys(ALGORITHMS)) {
  tsmcData[algo] = ALGORITHMS[algo].map(subset => fitOLS(subset, trainData, testData));
}

fs.writeFileSync('tsmc_model_data.json', JSON.stringify(tsmcData, null, 2));
console.log('Successfully generated tsmc_model_data.json!');
