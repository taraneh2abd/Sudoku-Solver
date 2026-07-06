const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");

input.onchange = () => {

    preview.src = URL.createObjectURL(input.files[0]);

};

document.getElementById("solveBtn").onclick = async () => {

    const file = input.files[0];

    if (!file) return;

    const form = new FormData();

    form.append("image", file);

    const res = await fetch("/upload", {
        method: "POST",
        body: form
    });

    const data = await res.json();

    drawBoard(data.original, "originalBoard", false);

    if (data.solved === "UNSOLVABLE") {

        document.getElementById("solvedBoard").innerHTML =
            `<div class="unsolved">
            ❌ This Sudoku is unsolvable
        </div>`;

    } else {

        drawBoard(data.solved, "solvedBoard", true);

    }

};

function drawBoard(board, id, solved) {

    const container = document.getElementById(id);

    container.innerHTML = "";

    const table = document.createElement("table");

    table.className = "sudoku";

    for (let i = 0; i < 9; i++) {

        const tr = document.createElement("tr");

        for (let j = 0; j < 9; j++) {

            const td = document.createElement("td");

            td.textContent = board[i][j] === 0 ? "" : board[i][j];

            if (solved)
                td.classList.add("solved");

            if (i % 3 === 0)
                td.style.borderTopWidth = "3px";

            if (j % 3 === 0)
                td.style.borderLeftWidth = "3px";

            if (i === 8)
                td.style.borderBottomWidth = "3px";

            if (j === 8)
                td.style.borderRightWidth = "3px";

            tr.appendChild(td);
        }

        table.appendChild(tr);
    }

    container.appendChild(table);

}