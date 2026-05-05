const apiMessage = document.querySelector("#api-message");

async function loadApiMessage() {
    try {
        const response = await fetch("http://127.0.0.1:8000/");

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();
        apiMessage.textContent = data.message;
    } catch (error) {
        apiMessage.textContent = "Не удалось получить сообщение от backend API";
        console.error(error);
    }
}

loadApiMessage();
