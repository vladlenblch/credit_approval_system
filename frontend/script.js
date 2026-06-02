const API_URL = "http://127.0.0.1:8000/predict";

const form = document.querySelector("#features-form");
const submitButton = document.querySelector("#submit-button");
const resultPanel = document.querySelector("#result-panel");
const resultStatus = document.querySelector("#result-status");
const resultContent = document.querySelector("#result-content");
const decisionValue = document.querySelector("#decision-value");
const probabilityValue = document.querySelector("#probability-value");
const thresholdValue = document.querySelector("#threshold-value");
const catboostScore = document.querySelector("#catboost-score");
const lightgbmScore = document.querySelector("#lightgbm-score");
const xgboostScore = document.querySelector("#xgboost-score");

function formatPercent(value) {
    return `${(value * 100).toFixed(2)}%`;
}

function setLoadingState() {
    submitButton.disabled = true;
    submitButton.textContent = "Расчет...";
    resultPanel.className = "result-panel";
    resultStatus.textContent = "Модель считает риск";
    resultContent.hidden = true;
}

function setErrorState(message) {
    submitButton.disabled = false;
    submitButton.textContent = "Рассчитать";
    resultPanel.className = "result-panel is-error";
    resultStatus.textContent = message;
    resultContent.hidden = true;
}

function setResultState(data) {
    const isApproved = data.approval_decision === "approve";

    submitButton.disabled = false;
    submitButton.textContent = "Рассчитать";
    resultPanel.className = `result-panel ${isApproved ? "is-approved" : "is-rejected"}`;
    resultStatus.textContent = isApproved ? "Кредит можно одобрить" : "Заявку лучше отклонить";
    resultContent.hidden = false;

    decisionValue.textContent = isApproved ? "Одобрить" : "Отклонить";
    probabilityValue.textContent = formatPercent(data.default_probability);
    thresholdValue.textContent = formatPercent(data.decision_threshold);
    catboostScore.textContent = formatPercent(data.model_scores.catboost);
    lightgbmScore.textContent = formatPercent(data.model_scores.lightgbm);
    xgboostScore.textContent = formatPercent(data.model_scores.xgboost);
}

function buildPayload() {
    const formData = new FormData(form);
    const payload = {};

    for (const [key, value] of formData.entries()) {
        payload[key] = value === "" ? "" : Number(value);
    }

    return payload;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setLoadingState();

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(buildPayload()),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail?.message || data.detail || `API returned ${response.status}`);
        }

        setResultState(data);
    } catch (error) {
        setErrorState("Не удалось получить ответ от модели");
        console.error(error);
    }
});
