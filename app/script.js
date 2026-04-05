const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const statusEl = document.getElementById("status");
const loadingEl = document.getElementById("loading");
const resultsEl = document.getElementById("results");
const uploadedPreviewEl = document.getElementById("uploaded-preview");
const matchedImageEl = document.getElementById("matched-image");
const productNameEl = document.getElementById("product-name");
const submitBtn = document.getElementById("submit-btn");
const defaultSubmitText = submitBtn.textContent;

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const selectedFile = imageInput.files?.[0];
  if (!selectedFile) {
    statusEl.textContent = "Please choose an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);

  resultsEl.classList.add("hidden");
  productNameEl.classList.add("hidden");
  productNameEl.textContent = "";
  matchedImageEl.removeAttribute("src");

  statusEl.textContent = "Sit tight while we look for a match!";
  loadingEl.classList.remove("hidden");
  submitBtn.disabled = true;
  submitBtn.textContent = "Searching...";

  const localPreviewUrl = URL.createObjectURL(selectedFile);
  uploadedPreviewEl.src = localPreviewUrl;

  try {
    const response = await fetch("/api/find-similar", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Upload failed.");
    }
    if (data.error) {
      throw new Error(data.error);
    }

    matchedImageEl.src = data.matched_image;
    if (data.product_name) {
      if (data.product_link) {
        productNameEl.textContent = "Product: ";
        const nameLink = document.createElement("a");
        nameLink.href = data.product_link;
        nameLink.target = "_blank";
        nameLink.rel = "noopener noreferrer";
        nameLink.textContent = data.product_name;
        productNameEl.appendChild(nameLink);
      } else {
        productNameEl.textContent = `Product: ${data.product_name}`;
      }
      productNameEl.classList.remove("hidden");
    } else {
      productNameEl.classList.add("hidden");
      productNameEl.textContent = "";
    }

    resultsEl.classList.remove("hidden");
    const scorePct = Number(data.match_score_pct);
    statusEl.textContent = Number.isFinite(scorePct)
      ? `Match Found (Confidence: ${scorePct.toFixed(2)}%)`
      : "Match Found!";
  } catch (error) {
    statusEl.textContent = error.message || "Something went wrong.";
    resultsEl.classList.add("hidden");
    productNameEl.classList.add("hidden");
  } finally {
    loadingEl.classList.add("hidden");
    submitBtn.disabled = false;
    submitBtn.textContent = defaultSubmitText;
  }
});
