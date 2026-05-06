const form = document.querySelector("#features-form");
const backendResult = document.querySelector("#backend-result");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    try {
        const response = await fetch("http://127.0.0.1:8000/check-features", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();
        backendResult.textContent = String(data.result);
    } catch (error) {
        backendResult.textContent = "Не удалось получить ответ от backend";
        console.error(error);
    }
});
