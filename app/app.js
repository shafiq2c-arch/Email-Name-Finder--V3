const API_BASE = window.location.origin;

// Selectors
const companyInput = document.getElementById('company');
const designationInput = document.getElementById('designation');
const searchBtn = document.getElementById('search-btn');
const singleResult = document.getElementById('single-result');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name');
const bulkBtn = document.getElementById('bulk-btn');
const bulkStatus = document.getElementById('bulk-status');

const resultsContainer = document.getElementById('results-container');
const resultsTable = document.getElementById('results-table').querySelector('tbody');
const downloadBtn = document.getElementById('download-btn');

let selectedFile = null;
let bulkResultsBlob = null;

// --- Single Search ---
searchBtn.addEventListener('click', async () => {
    const company = companyInput.value.trim();
    const designation = designationInput.value.trim();

    if (!company || !designation) {
        alert("Please enter both company and designation.");
        return;
    }

    console.log("Searching for:", company, designation);
    setLoading(searchBtn, true);
    singleResult.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ company, designation })
        });

        console.log("Response status:", response.status);
        if (!response.ok) throw new Error("Search failed on server.");

        const data = await response.json();
        console.log("Search result:", data);
        addResultToTable(data);
        showSingleResult(data);
    } catch (err) {
        console.error("Search error:", err);
        alert(err.message);
    } finally {
        setLoading(searchBtn, false);
    }
});

function showSingleResult(data) {
    singleResult.innerHTML = `
        <div class="result-name">${data.name}</div>
        <div class="result-company">${data.company} - ${data.designation}</div>
        <div class="result-designation">
            <span class="badge ${getBadgeClass(data.confidence)}">${data.confidence}% Confidence</span>
            <span style="margin-left: 10px;">${data.status}</span>
        </div>
        ${data.source !== 'N/A' ? `<div class="result-source"><a href="${data.source}" target="_blank" class="source-link">View Source</a></div>` : ''}
    `;
    singleResult.classList.remove('hidden');
}

// --- Bulk Search ---
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file || !file.name.endsWith('.csv')) {
        alert("Please select a valid CSV file.");
        return;
    }
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    bulkBtn.disabled = false;
}

bulkBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    setLoading(bulkBtn, true);
    bulkStatus.textContent = "Processing bulk list... this may take a while.";
    bulkStatus.classList.remove('hidden');

    try {
        const response = await fetch(`${API_BASE}/bulk-search`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Bulk search failed.");

        bulkResultsBlob = await response.blob();

        // Note: For simplicity in vanilla JS, we don't display all bulk results in the table 
        // to avoid freezing the UI, but we provide the download link.
        bulkStatus.textContent = "Bulk processing complete!";
        downloadBtn.classList.remove('hidden');
        resultsContainer.classList.remove('hidden');
    } catch (err) {
        bulkStatus.textContent = "Error: " + err.message;
    } finally {
        setLoading(bulkBtn, false);
    }
});

downloadBtn.addEventListener('click', () => {
    if (!bulkResultsBlob) return;
    const url = window.URL.createObjectURL(bulkResultsBlob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `results_${selectedFile.name}`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
});

// --- Utilities ---
function setLoading(btn, isLoading) {
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    if (isLoading) {
        text.classList.add('hidden');
        loader.classList.remove('hidden');
        btn.disabled = true;
    } else {
        text.classList.remove('hidden');
        loader.classList.add('hidden');
        btn.disabled = false;
    }
}

function addResultToTable(data) {
    resultsContainer.classList.remove('hidden');
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${data.company}</td>
        <td>${data.designation}</td>
        <td>${data.name}</td>
        <td><span class="badge ${getBadgeClass(data.confidence)}">${data.confidence}%</span></td>
        <td>${data.source !== 'N/A' ? `<a href="${data.source}" target="_blank" class="source-link">Link</a>` : 'N/A'}</td>
    `;
    resultsTable.prepend(row);
}

function getBadgeClass(conf) {
    if (conf >= 80) return 'badge-high';
    if (conf >= 50) return 'badge-mid';
    return 'badge-low';
}
