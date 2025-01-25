
// // Wait for the DOM to be fully loaded before running the script
// document.addEventListener("DOMContentLoaded", function () {
//   // Get the form element and the result display element
//   const form = document.querySelector("#predictionForm");
//   const resultDiv = document.querySelector("#result");
//
//   // Check if the form exists in the DOM
//   if (form) {
//     // Add an event listener for form submission
//     form.addEventListener("submit", async function (event) {
//       event.preventDefault(); // Prevent the default form submission behavior
//
//       // Collect input data from the form
//       const formData = new FormData(form);
//       const data = Object.fromEntries(formData.entries());
//
//       try {
//         // Send the data to the server via fetch
//         const response = await fetch("/predict", {
//           method: "POST",
//           headers: {
//             "Content-Type": "application/json",
//           },
//           body: JSON.stringify(data),
//         });
//
//         // Parse the JSON response
//         const result = await response.json();
//
//         // Display the result on the page
//         if (resultDiv) {
//           resultDiv.textContent = `Prediction: ${result.prediction}`;
//         }
//       } catch (error) {
//         console.error("Error occurred during prediction:", error);
//         if (resultDiv) {
//           resultDiv.textContent = "An error occurred. Please try again.";
//         }
//       }
//     });
//   } else {
//     console.error("Form with id 'predictionForm' not found in the DOM.");
//   }
// });

// document.addEventListener("DOMContentLoaded", function () {
//     const form = document.querySelector("#predictionForm");
//     const resultDiv = document.querySelector("#result");
//
//     if (form) {
//         form.addEventListener("submit", async function (event) {
//             event.preventDefault();
//
//             const formData = new FormData(form);
//             const data = Object.fromEntries(formData.entries());
//
//             try {
//                 const response = await fetch("/predict", {
//                     method: "POST",
//                     headers: {
//                         "Content-Type": "application/json",
//                     },
//                     body: JSON.stringify(data),
//                 });
//
//                 // Check if the response is OK (status code 200)
//                 if (!response.ok) {
//                     throw new Error(`HTTP error! status: ${response.status}`);
//                 }
//
//                 // Parse the JSON response
//                 const result = await response.json();
//
//                 if (resultDiv) {
//                     resultDiv.textContent = `Prediction: ${result.prediction}`;
//                 }
//             } catch (error) {
//                 console.error("Error occurred during prediction:", error);
//                 if (resultDiv) {
//                     resultDiv.textContent = "An error occurred. Please try again.";
//                 }
//             }
//         });
//     }
// });

document.getElementById("showLeaderboard").addEventListener("click", async () => {
    console.log("Leaderboard button clicked"); // Check if this prints to the console

    try {
        const response = await fetch("/leaderboard");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const leaderboard = await response.json();

        console.log("Leaderboard data:", leaderboard); // Check if the data is received

        const leaderboardDiv = document.getElementById("leaderboard");
        leaderboardDiv.innerHTML = `
            <h3>Leaderboard</h3>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Username</th>
                        <th>Efficiency</th>
                        <th>Prediction</th>
                    </tr>
                </thead>
                <tbody>
                    ${leaderboard.map(row => `
                        <tr>
                            <td>${row.username}</td>
                            <td>${row.efficiency.toFixed(2)}</td>
                            <td>${row.prediction}</td>
                        </tr>
                    `).join("")}
                </tbody>
            </table>
        `;
    } catch (error) {
        console.error("Error fetching leaderboard:", error);
    }
});
