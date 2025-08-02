function analyzeText() {
    let text = document.getElementById("userInput").value;
    
    fetch("https://492d7a106190.ngrok-free.app/detect", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerHTML = `Error: ${data.error}`;
        } else {
            let bias = data.final_bias ? "⚠️ Biased" : "✅ Not Biased";
            document.getElementById("result").innerHTML = `Result: ${bias}`;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerHTML = `Error analyzing text: ${error.message}`;
    });
}
