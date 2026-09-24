(() => {
  const $ = id => document.getElementById(id);
  let requestId = null;
  const fail = message => { $('lab-alert').textContent = message; $('lab-alert').hidden = !message; };

  $('lab-image').onchange = () => {
    const file = $('lab-image').files[0];
    if (!file) return;
    $('lab-preview').src = URL.createObjectURL(file);
    $('lab-preview').hidden = false;
    $('lab-pick').hidden = true;
    $('lab-output').hidden = true;
    requestId = null;
    fail('');
  };

  $('lab-generate').onclick = async () => {
    fail('');
    const data = new FormData();
    const file = $('lab-image').files[0];
    if (file) data.append('image', file);
    data.append('prompt', $('lab-prompt').value);
    data.append('strength', document.querySelector('input[name=strength]:checked').value);
    $('lab-generate').disabled = true;
    try {
      const response = await fetch('/api/prompt-perturb', {method: 'POST', body: data});
      const body = await response.json();
      if (!response.ok) throw Error(body.error || 'Generation failed.');
      requestId = body.request_id;
      $('original-image').src = body.original_image_url;
      $('modified-image').src = body.modified_image_url;
      $('difference-image').src = body.difference_image_url;
      $('lab-output').hidden = false;
      $('lab-results').hidden = true;
    } catch (error) { fail(error.message); }
    finally { $('lab-generate').disabled = false; }
  };

  const featureList = features => features.map(item => `<div class="feature-item"><span>${item.feature_name}<small>${item.feature_domain}</small></span><strong>${Number(item.contribution).toExponential(2)}</strong></div>`).join('');

  $('scan-both').onclick = async () => {
    if (!requestId) return;
    fail('');
    $('scan-both').disabled = true;
    try {
      const response = await fetch('/api/compare-scan', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({request_id: requestId, model_id: $('lab-model').value}),
      });
      const body = await response.json();
      if (!response.ok) throw Error(body.error || 'Scan failed.');
      const original = body.original, modified = body.modified, pair = body.pair_detection;
      const pairBox = $('pair-result');
      $('risk-delta').textContent = `${body.risk_delta >= 0 ? '+' : ''}${body.risk_delta.toFixed(1)} percentage points`;
      $('interpretation').textContent = body.interpretation;
      if (pair && pairBox) {
        pairBox.hidden = false;
        pairBox.classList.toggle('detected', pair.detected);
        $('pair-status').textContent = modified.lab_assessment;
        $('pair-scope').textContent = pair.detected ? 'Prompt-conditioned modification detected by comparison with the original image.' : 'No prompt-conditioned modification was detected by comparison with the original image.';
        $('pair-confidence').textContent = `${pair.probability.toFixed(1)}% confidence`;
      } else if (pairBox) pairBox.hidden = true;
      const assessment = modified.lab_assessment || modified.status;
      $('comparison-body').innerHTML = [
        ['Status', original.lab_assessment || original.status, assessment],
        ['Selected Model', original.model_name, modified.model_name],
        ['Single-image Attack Score', `${original.attack_probability.toFixed(1)}%`, `${modified.attack_probability.toFixed(1)}%`],
        ['Single-image Clean Score', `${original.clean_probability.toFixed(1)}%`, `${modified.clean_probability.toFixed(1)}%`],
        ['Pair Detection Confidence', '—', pair ? `${pair.probability.toFixed(1)}%` : 'Unavailable'],
      ].map((row, index) => `<tr><th>${row[0]}</th><td>${row[1]}</td><td class="${index === 0 && assessment !== 'VERIFIED/CLEAN' ? 'status-concern' : ''}">${row[2]}</td></tr>`).join('');
      $('original-features').innerHTML = featureList(original.top_features);
      $('modified-features').innerHTML = featureList(modified.top_features);
      $('delta-body').innerHTML = body.feature_deltas.map(item => `<tr><td>${item.feature_name}</td><td>${item.feature_domain}</td><td>${item.original_value.toPrecision(4)}</td><td>${item.modified_value.toPrecision(4)}</td><td>${item.absolute_delta.toPrecision(4)}</td><td>${item.relative_delta === null ? '∞' : (100 * item.relative_delta).toFixed(1) + '%'}</td></tr>`).join('');
      $('lab-results').hidden = false;
    } catch (error) { fail(error.message); }
    finally { $('scan-both').disabled = false; }
  };

  $('lab-reset').onclick = () => {
    requestId = null; $('lab-image').value = ''; $('lab-prompt').value = '';
    document.querySelector('input[value=low]').checked = true;
    $('lab-preview').removeAttribute('src'); $('lab-preview').hidden = true;
    $('lab-pick').hidden = false; $('lab-output').hidden = true; $('lab-results').hidden = true; fail('');
  };
})();
