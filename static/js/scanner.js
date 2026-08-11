(() => {
  const form = document.querySelector('#scan-form');
  if (!form) return;
  const input = document.querySelector('#image-input');
  const dropZone = document.querySelector('#drop-zone');
  const preview = document.querySelector('#preview');
  const copy = document.querySelector('#drop-copy');
  const fileName = document.querySelector('#file-name');
  const button = document.querySelector('#scan-button');
  const label = button.querySelector('.button-label');
  const spinner = button.querySelector('.spinner');
  const allowedExtensions = new Set(['jpg', 'jpeg', 'png']);
  const maxBytes = Number(input.dataset.maxBytes || 0);

  const showFile = (file) => {
    if (!file) return;
    const extension = file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.has(extension)) {
      input.setCustomValidity('Choose a JPG, JPEG, or PNG image.');
      fileName.textContent = 'Unsupported file type';
      return;
    }
    if (maxBytes && file.size > maxBytes) {
      input.setCustomValidity('The selected image exceeds the upload size limit.');
      fileName.textContent = 'Image exceeds the upload size limit';
      return;
    }
    input.setCustomValidity('');
    fileName.textContent = `${file.name} · ${(file.size / 1048576).toFixed(2)} MB`;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    copy.hidden = true;
  };
  input.addEventListener('change', () => showFile(input.files[0]));
  ['dragenter', 'dragover'].forEach(event => dropZone.addEventListener(event, e => { e.preventDefault(); dropZone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(event => dropZone.addEventListener(event, e => { e.preventDefault(); dropZone.classList.remove('dragging'); }));
  dropZone.addEventListener('drop', event => {
    if (event.dataTransfer.files.length) {
      const transfer = new DataTransfer();
      transfer.items.add(event.dataTransfer.files[0]);
      input.files = transfer.files;
      showFile(input.files[0]);
    }
  });
  form.addEventListener('submit', () => {
    button.disabled = true;
    label.textContent = 'Analyzing image…';
    spinner.hidden = false;
  });
})();
