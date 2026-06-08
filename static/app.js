/* ────────────────────────────────────────────────────────────
   AI Video Transcriber · app.js
   ──────────────────────────────────────────────────────────── */

class VideoTranscriber {
  constructor() {
    this.currentTaskId  = null;
    this.eventSource    = null;
    this.apiBase        = '/api';
    this.currentLang    = 'en';
    this.currentStatistics = null;

    /* Smart progress simulation */
    this.sp = {
      enabled: false, current: 0, target: 15,
      lastServer: 0, interval: null, startTime: null, stage: 'preparing'
    };

    this.i18n = {
      en: {
        title:                   'AI Video Transcriber',
        subtitle:                'Supports automatic transcription and AI summary for 30+ platforms',
        video_url_placeholder:   'Paste YouTube, Tiktok, Bilibili or other platform video URLs...',
        start_transcription:     'Transcribe',
        ai_settings:             'AI Settings',
        model_base_url:          'Model API Base URL',
        model_base_url_placeholder: 'https://openrouter.ai/api/v1',
        api_key:                 'API Key',
        api_key_placeholder:     'sk-...',
        fetch_models:            'Fetch',
        model_select:            'Model',
        model_default:           '— use server default —',
        summary_language:        'Summary Language',
        processing_progress:     'Processing',
        preparing:               'Preparing…',
        transcript_text:         'Transcript',
        intelligent_summary:     'AI Summary',
        translation:             'Translation',
        statistics:              'Statistics',
        download_transcript:     'Transcript',
        download_translation:    'Translation',
        download_summary:        'Summary',
        empty_hint:              'Paste a video URL or drop a file above and let AI do the heavy lifting.',
        footer_text:             'This tool is part of <a href="https://sipsip.ai" target="_blank" style="color:var(--accent-text);text-decoration:none;">sipsip.ai</a> — distill anything and get daily AI briefs from your favorite creators',
        processing:              'Processing…',
        downloading_video:       'Downloading audio…',
        parsing_video:           'Parsing video info…',
        transcribing_audio:      'Transcribing audio…',
        optimizing_transcript:   'Optimizing transcript…',
        generating_summary:      'Generating summary…',
        detecting_subtitles:     'Detecting subtitles…',
        subtitle_found:          'Subtitles found! Processing text…',
        no_subtitle:             'No subtitles found, downloading audio…',
        mode_subtitle:           '⚡ Subtitle',
        mode_whisper:            '🎙 Whisper',
        completed:               'Done!',
        error_invalid_url:       'Please enter a valid video URL',
        error_processing_failed: 'Processing failed: ',
        error_no_download:       'No file available for download',
        error_download_failed:   'Download failed: ',
        fetching_models:         'Fetching models…',
        models_loaded:           (n) => `${n} models loaded`,
        models_error:            'Failed to fetch models',
        upload_or:               'or drop your files',
        upload_formats:          '.mp3 · .mp4 · .wav · .m4a · .webm · .mkv · .ogg · .flac',
        upload_files_btn:        'Upload files',
        error_upload_type:       'Unsupported file type',
        error_upload_empty:      'File is empty',
        saved_artifacts:         'Saved results',
        statistics_unavailable:  'Statistics are not available for this saved result.',
        stat_processing_time:    'Processing time',
        stat_input:              'Input',
        stat_source_type:        'Source type',
        stat_extraction:         'Extraction',
        stat_started:            'Started',
        stat_finished:           'Finished',
        stat_detected_language:  'Detected language',
        stat_summary_language:   'Summary language',
        stat_translation:        'Translation',
        stat_model:              'AI model',
        stat_raw_transcript:     'Raw transcript',
        stat_optimized_transcript: 'Optimized transcript',
        stat_summary:            'Summary',
        stat_chars:              'chars',
        stat_words:              'words',
        stat_lines:              'lines',
        stat_yes:                'Yes',
        stat_no:                 'No',
        stat_seconds:            'sec',
        stat_minutes:            'min',
      },
      zh: {}
    };

    this._initElements();
    this._bindEvents();
    this._loadSettings();
    this._switchLang('en');
    this._loadSavedArtifacts();
  }

  /* ── Elements ─────────────────────────────────────────── */
  _initElements() {
    this.form               = document.getElementById('videoForm');
    this.videoUrlInput      = document.getElementById('videoUrl');
    this.submitBtn          = document.getElementById('submitBtn');
    this.summaryLangSel     = document.getElementById('summaryLanguage');
    this.langToggle         = document.getElementById('langToggle');
    this.langText           = document.getElementById('langText');
    this.errorBanner        = document.getElementById('errorBanner');
    this.errorMsg           = document.getElementById('errorMsg');
    this.emptyState         = document.getElementById('emptyState');
    this.progressPanel      = document.getElementById('progressPanel');
    this.modeBadge          = document.getElementById('modeBadge');
    this.progressStatus     = document.getElementById('progressStatus');
    this.progressFill       = document.getElementById('progressFill');
    this.progressMessage    = document.getElementById('progressMessage');
    this.resultsPanel       = document.getElementById('resultsPanel');
    this.scriptContent      = document.getElementById('scriptContent');
    this.summaryContent     = document.getElementById('summaryContent');
    this.translationContent = document.getElementById('translationContent');
    this.statisticsContent  = document.getElementById('statisticsContent');
    this.dlScript           = document.getElementById('downloadScript');
    this.dlTranslation      = document.getElementById('downloadTranslation');
    this.dlSummary          = document.getElementById('downloadSummary');
    this.translationTabBtn  = document.getElementById('translationTabBtn');
    this.tabBtns            = document.querySelectorAll('.tab-btn');
    this.tabPanes           = document.querySelectorAll('.tab-pane');
    // settings
    this.settingsToggle     = document.getElementById('settingsToggle');
    this.settingsBody       = document.getElementById('settingsBody');
    this.modelBaseUrl       = document.getElementById('modelBaseUrl');
    this.apiKeyInput        = document.getElementById('apiKeyInput');
    this.fetchModelsBtn     = document.getElementById('fetchModelsBtn');
    this.fetchStatus        = document.getElementById('fetchStatus');
    this.modelSelect        = document.getElementById('modelSelect');
    this.fetchIcon          = document.getElementById('fetchIcon');
    this.uploadZone         = document.getElementById('uploadZone');
    this.uploadPickBtn      = document.getElementById('uploadPickBtn');
    this.fileInput          = document.getElementById('fileInput');
    this.historyPanel       = document.getElementById('historyPanel');
    this.historyList        = document.getElementById('historyList');
    this._allowedUploadExts = new Set(['.txt', '.mp3', '.mp4', '.m4a', '.wav', '.webm', '.mkv', '.ogg', '.flac']);
  }

  /* ── Events ───────────────────────────────────────────── */
  _bindEvents() {
    this.form.addEventListener('submit', (e) => { e.preventDefault(); this._startTranscription(); });

    this.langToggle.addEventListener('click', () => {
      this._switchLang(this.currentLang === 'en' ? 'zh' : 'en');
    });

    // Settings toggle
    this.settingsToggle.addEventListener('click', () => {
      const open = this.settingsBody.classList.toggle('open');
      this.settingsToggle.classList.toggle('open', open);
    });

    // Fetch models
    this.fetchModelsBtn.addEventListener('click', () => this._fetchModels());

    // Auto-fetch when both fields filled (debounced)
    const debouncedFetch = this._debounce(() => {
      if (this.modelBaseUrl.value.trim() && this.apiKeyInput.value.trim()) this._fetchModels();
    }, 900);
    this.modelBaseUrl.addEventListener('input', debouncedFetch);
    this.apiKeyInput.addEventListener('input', debouncedFetch);

    // Persist settings
    [this.modelBaseUrl, this.apiKeyInput, this.modelSelect, this.summaryLangSel].forEach(el => {
      el.addEventListener('change', () => this._saveSettings());
    });

    // Tabs
    this.tabBtns.forEach(btn => {
      btn.addEventListener('click', () => this._switchTab(btn.dataset.tab));
    });

    // Downloads
    this.dlScript.addEventListener('click',      () => this._downloadFile('script'));
    this.dlTranslation.addEventListener('click', () => this._downloadFile('translation'));
    this.dlSummary.addEventListener('click',     () => this._downloadFile('summary'));

    if (this.uploadPickBtn && this.fileInput && this.uploadZone) {
      this.uploadPickBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.fileInput.click();
      });
      this.uploadZone.addEventListener('click', (e) => {
        if (e.target === this.uploadPickBtn || this.uploadPickBtn.contains(e.target)) return;
        this.fileInput.click();
      });
      this.fileInput.addEventListener('change', () => {
        const f = this.fileInput.files && this.fileInput.files[0];
        this.fileInput.value = '';
        if (f) this._startFileUpload(f);
      });
      ['dragenter', 'dragover'].forEach((ev) => {
        this.uploadZone.addEventListener(ev, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.uploadZone.classList.add('dragover');
        });
      });
      this.uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        if (!this.uploadZone.contains(e.relatedTarget)) {
          this.uploadZone.classList.remove('dragover');
        }
      });
      this.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');
        const f = e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) this._startFileUpload(f);
      });
    }

    if (this.historyList) {
      this.historyList.addEventListener('click', (e) => {
        const item = e.target.closest('[data-task-id]');
        if (item) this._openSavedTask(item.dataset.taskId);
      });
    }
  }

  /* ── i18n ─────────────────────────────────────────────── */
  t(key) { return this.i18n[this.currentLang][key] || this.i18n['en'][key] || key; }

  _switchLang(lang) {
    this.currentLang = lang;
    this.langText.textContent = lang === 'en' ? 'English' : 'Chinese';
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = this.t('title');

    document.querySelectorAll('[data-i18n]').forEach(el => {
      const v = this.t(el.dataset.i18n);
      if (typeof v === 'string') {
        // The footer allows HTML; all other keys use textContent.
        if (el.dataset.i18n === 'footer_text') el.innerHTML = v;
        else el.textContent = v;
      }
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const v = this.t(el.dataset.i18nPlaceholder);
      if (typeof v === 'string') el.placeholder = v;
    });
    if (this.currentStatistics) this._renderStatistics(this.currentStatistics);
  }

  /* ── Settings persistence ─────────────────────────────── */
  _saveSettings() {
    const s = {
      baseUrl:  this.modelBaseUrl.value,
      apiKey:   this.apiKeyInput.value,
      model:    this.modelSelect.value,
      summaryLang: this.summaryLangSel.value,
    };
    try { localStorage.setItem('vt_settings', JSON.stringify(s)); } catch (_) {}
  }

  _loadSettings() {
    try {
      const raw = localStorage.getItem('vt_settings');
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.baseUrl)     this.modelBaseUrl.value = s.baseUrl;
      if (s.apiKey)      this.apiKeyInput.value  = s.apiKey;
      if (s.summaryLang) this.summaryLangSel.value = s.summaryLang;
      // Model options will be restored after fetching
      this._savedModel = s.model || '';

      // Auto-open settings if credentials were saved
      if (s.baseUrl || s.apiKey) {
        this.settingsBody.classList.add('open');
        this.settingsToggle.classList.add('open');
        // Attempt to re-fetch model list silently
        if (s.baseUrl && s.apiKey) {
          setTimeout(() => this._fetchModels(true), 400);
        }
      }
    } catch (_) {}
  }

  async _loadSavedArtifacts() {
    if (!this.historyPanel || !this.historyList) return;
    try {
      const resp = await fetch(`${this.apiBase}/artifacts`);
      if (!resp.ok) return;
      const data = await resp.json();
      const items = data.items || [];
      this.historyList.innerHTML = '';
      if (!items.length) {
        this.historyPanel.classList.remove('show');
        return;
      }

      items.forEach(item => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'history-item';
        btn.dataset.taskId = item.task_id;
        btn.innerHTML = `<i class="fas fa-file-alt"></i><span class="history-name"></span>`;
        btn.querySelector('.history-name').textContent = item.input_name || item.video_title || item.task_id;
        this.historyList.appendChild(btn);
      });
      this.historyPanel.classList.add('show');
    } catch (_) {}
  }

  async _openSavedTask(taskId) {
    try {
      const resp = await fetch(`${this.apiBase}/task-status/${encodeURIComponent(taskId)}`);
      if (!resp.ok) throw new Error('Failed to load saved result');
      const task = await resp.json();
      this.currentTaskId = taskId;
      this._stopSP();
      this._stopSSE();
      this._setLoading(false);
      this._hideProgress();
      this._hideError();
      this._showResults(task.script, task.summary, task.video_title, task.translation, task.detected_language, task.summary_language, task.statistics);
    } catch (e) {
      this._showError(this.t('error_processing_failed') + e.message);
    }
  }

  /* ── Fetch models ─────────────────────────────────────── */
  async _fetchModels(silent = false) {
    const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
    const apiKey  = this.apiKeyInput.value.trim();

    if (!baseUrl || !apiKey) {
      if (!silent) this._setFetchStatus('err', this.t('api_key') + ' & URL required');
      return;
    }

    this.fetchModelsBtn.disabled = true;
    this.fetchIcon.className = 'fas fa-spinner fa-spin';
    if (!silent) this._setFetchStatus('', this.t('fetching_models'));

    try {
      const fd = new FormData();
      fd.append('base_url', baseUrl);
      fd.append('api_key',  apiKey);

      const resp = await fetch(`${this.apiBase}/models`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      const models = data.data || data.models || [];

      // Rebuild select options
      this.modelSelect.innerHTML = `<option value="">${this.t('model_default')}</option>`;
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name || m.id;
        this.modelSelect.appendChild(opt);
      });

      // Restore previously selected model
      if (this._savedModel) {
        this.modelSelect.value = this._savedModel;
        this._savedModel = '';
      }

      this._setFetchStatus('ok', typeof this.t('models_loaded') === 'function'
        ? this.t('models_loaded')(models.length)
        : `${models.length} models`);

    } catch (e) {
      console.warn('Model fetch error:', e);
      this._setFetchStatus('err', this.t('models_error') + ': ' + e.message);
    } finally {
      this.fetchModelsBtn.disabled = false;
      this.fetchIcon.className = 'fas fa-sync-alt';
    }
  }

  _setFetchStatus(cls, msg) {
    this.fetchStatus.className = 'fetch-status' + (cls ? ` ${cls}` : '');
    this.fetchStatus.textContent = msg;
  }

  /* ── Transcription ────────────────────────────────────── */
  async _startTranscription() {
    if (this.submitBtn.disabled) return;

    const url     = this.videoUrlInput.value.trim();
    const sumLang = this.summaryLangSel.value;

    if (!url) { this._showError(this.t('error_invalid_url')); return; }

    this._setLoading(true);
    this._hideError();
    this._showProgress();

    try {
      const fd = new FormData();
      fd.append('url',              url);
      fd.append('summary_language', sumLang);

      const apiKey  = this.apiKeyInput.value.trim();
      const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
      const modelId = this.modelSelect.value;
      if (apiKey)  fd.append('api_key',       apiKey);
      if (baseUrl) fd.append('model_base_url', baseUrl);
      if (modelId) fd.append('model_id',       modelId);

      const resp = await fetch(`${this.apiBase}/process-video`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || 'Request failed');
      }

      const data = await resp.json();
      this.currentTaskId = data.task_id;

      this._initSP();
      this._updateProgress(5, this.t('preparing'), true);
      this._startSSE();
      this._saveSettings();

    } catch (err) {
      this._showError(this.t('error_processing_failed') + err.message);
      this._setLoading(false);
      this._hideProgress();
    }
  }

  async _startFileUpload(file) {
    if (this.submitBtn.disabled) return;

    const parts = (file.name || '').split('.');
    const ext = parts.length > 1 ? ('.' + parts.pop().toLowerCase()) : '';
    if (!this._allowedUploadExts.has(ext)) {
      this._showError(this.t('error_upload_type'));
      return;
    }
    if (!file.size) {
      this._showError(this.t('error_upload_empty'));
      return;
    }
    this._setLoading(true);
    this._hideError();
    this._showProgress();

    const sumLang = this.summaryLangSel.value;
    try {
      const fd = new FormData();
      fd.append('file', file, file.name);
      fd.append('summary_language', sumLang);

      const apiKey  = this.apiKeyInput.value.trim();
      const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
      const modelId = this.modelSelect.value;
      if (apiKey)  fd.append('api_key',       apiKey);
      if (baseUrl) fd.append('model_base_url', baseUrl);
      if (modelId) fd.append('model_id',       modelId);

      const resp = await fetch(`${this.apiBase}/process-video`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        const d = err.detail;
        const msg = typeof d === 'string'
          ? d
          : (Array.isArray(d) && d[0] && (d[0].msg || d[0].message))
            || `HTTP ${resp.status}`;
        throw new Error(msg);
      }

      const data = await resp.json();
      this.currentTaskId = data.task_id;

      this._initSP();
      this._updateProgress(5, this.t('preparing'), true);
      this._startSSE();
      this._saveSettings();

    } catch (err) {
      this._showError(this.t('error_processing_failed') + err.message);
      this._setLoading(false);
      this._hideProgress();
    }
  }

  /* ── SSE ──────────────────────────────────────────────── */
  _startSSE() {
    if (!this.currentTaskId) return;
    this.eventSource = new EventSource(`${this.apiBase}/task-stream/${this.currentTaskId}`);

    this.eventSource.onmessage = (ev) => {
      try {
        const task = JSON.parse(ev.data);
        if (task.type === 'heartbeat') return;

        this._updateProgress(task.progress, task.message, true);

        if (task.status === 'completed') {
          this._stopSP(); this._stopSSE(); this._setLoading(false); this._hideProgress();
          this._showResults(task.script, task.summary, task.video_title, task.translation, task.detected_language, task.summary_language, task.statistics);
          this._loadSavedArtifacts();
        } else if (task.status === 'error') {
          this._stopSP(); this._stopSSE(); this._setLoading(false); this._hideProgress();
          this._showError(task.error || 'Processing error');
        }
      } catch (_) {}
    };

    this.eventSource.onerror = async () => {
      this._stopSSE();
      try {
        if (this.currentTaskId) {
          const r = await fetch(`${this.apiBase}/task-status/${this.currentTaskId}`);
          if (r.ok) {
            const task = await r.json();
            if (task?.status === 'completed') {
              this._stopSP(); this._setLoading(false); this._hideProgress();
              this._showResults(task.script, task.summary, task.video_title, task.translation, task.detected_language, task.summary_language, task.statistics);
              this._loadSavedArtifacts();
              return;
            }
          }
        }
      } catch (_) {}
      this._showError(this.t('error_processing_failed') + 'SSE disconnected');
      this._setLoading(false);
    };
  }

  _stopSSE() {
    if (this.eventSource) { this.eventSource.close(); this.eventSource = null; }
  }

  /* ── Progress ─────────────────────────────────────────── */
  _updateProgress(pct, msg, fromServer = false) {
    if (fromServer) {
      this._stopSP();
      this.sp.lastServer = pct;
      this.sp.current    = pct;
      this._renderProgress(pct, msg);
      this._updateStage(pct, msg);
      this._startSP();
    } else {
      this._renderProgress(pct, msg);
    }
  }

  _updateStage(pct, msg) {
    const m = (msg || '').toLowerCase();

    // ── Subtitle path (fast) ──────────────────────────────────
    if (m.includes('subtitles found') || m.includes('subtitle found')) {
      this.sp.stage = 'subtitle_found';
      this.sp.target = 55;
      this._setModeBadge('subtitle');
    }
    // ── No subtitles -> audio download path (slow) ────────────
    else if (m.includes('no subtitles') || m.includes('no subtitle') || m.includes('downloading video audio') || m.includes('downloading audio')) {
      this.sp.stage = 'downloading';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if ((m.includes('read') || m.includes('reading')) && m.includes('text')) {
      this.sp.stage = 'parsing';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if (m.includes('converting audio') || m.includes('preparing transcription')) {
      this.sp.stage = 'downloading';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if (m.includes('upload')) {
      this.sp.stage = 'preparing';
      this.sp.target = 40;
    }
    // ── Generic subtitle detection ────────────────────────────
    else if (m.includes('detect') && m.includes('subtitle')) {
      this.sp.stage = 'subtitle';
      this.sp.target = 40;
    }
    // ── Other stages ──────────────────────────────────────────
    else if (m.includes('pars'))                                           { this.sp.stage = 'parsing';       this.sp.target = 60; }
    else if (m.includes('download') || m.includes('converting audio'))     { this.sp.stage = 'downloading';   this.sp.target = 60; }
    else if (m.includes('transcrib') || m.includes('whisper'))             { this.sp.stage = 'transcribing';  this.sp.target = 80; }
    else if (m.includes('optimiz'))                                        { this.sp.stage = 'optimizing';    this.sp.target = 90; }
    else if (m.includes('summary'))                                        { this.sp.stage = 'summarizing';   this.sp.target = 95; }
    else if (m.includes('complet'))                                        { this.sp.stage = 'completed';     this.sp.target = 100; }

    if (pct >= this.sp.target) this.sp.target = Math.min(pct + 8, 99);
  }

  _setModeBadge(mode) {
    if (!this.modeBadge) return;
    if (mode === 'subtitle') {
      this.modeBadge.textContent  = this.t('mode_subtitle');
      this.modeBadge.className    = 'mode-badge subtitle';
      this.modeBadge.style.display = 'inline-block';
      if (this.progressFill) this.progressFill.classList.add('subtitle-mode');
    } else if (mode === 'whisper') {
      this.modeBadge.textContent  = this.t('mode_whisper');
      this.modeBadge.className    = 'mode-badge whisper';
      this.modeBadge.style.display = 'inline-block';
      if (this.progressFill) this.progressFill.classList.remove('subtitle-mode');
    }
  }

  _initSP() {
    this.sp.enabled = false; this.sp.current = 0; this.sp.target = 15;
    this.sp.lastServer = 0;  this.sp.startTime = Date.now(); this.sp.stage = 'preparing';
  }
  _startSP() {
    if (this.sp.interval) clearInterval(this.sp.interval);
    this.sp.enabled   = true;
    this.sp.startTime = this.sp.startTime || Date.now();
    this.sp.interval  = setInterval(() => this._tickSP(), 500);
  }
  _stopSP() {
    if (this.sp.interval) { clearInterval(this.sp.interval); this.sp.interval = null; }
    this.sp.enabled = false;
  }
  _tickSP() {
    if (!this.sp.enabled || this.sp.current >= this.sp.target) return;
    const speeds = { subtitle: .5, parsing: .3, downloading: .18, transcribing: .14, optimizing: .22, summarizing: .28 };
    let inc = speeds[this.sp.stage] || .2;
    const remaining = this.sp.target - this.sp.current;
    if (remaining < 5) inc *= .3;
    const next = Math.min(this.sp.current + inc, this.sp.target);
    if (next > this.sp.current) {
      this.sp.current = next;
      this._renderProgress(next, this._stageMsg());
    }
  }
  _stageMsg() {
    const map = {
      subtitle:       this.t('detecting_subtitles'),
      subtitle_found: this.t('subtitle_found'),
      downloading:    this.t('downloading_video'),
      parsing:        this.t('parsing_video'),
      transcribing:   this.t('transcribing_audio'),
      optimizing:     this.t('optimizing_transcript'),
      summarizing:    this.t('generating_summary'),
      completed:      this.t('completed'),
    };
    return map[this.sp.stage] || this.t('processing');
  }

  _renderProgress(pct, msg) {
    const p = Math.round(pct * 10) / 10;
    this.progressStatus.textContent = `${p}%`;
    this.progressFill.style.width   = `${p}%`;

    // Translate common server messages — more specific checks first
    const m = (msg || '').toLowerCase();
    let label = msg;
    // ── Subtitle path ──────────────────────────────────────────
    if      (m.includes('subtitles found') || m.includes('subtitle found')) label = this.t('subtitle_found');
    else if (m.includes('no subtitles') || m.includes('no subtitle'))       label = this.t('no_subtitle');
    else if (m.includes('detect') && m.includes('subtitle'))                label = this.t('detecting_subtitles');
    // ── Audio / Whisper path ────────────────────────────────────
    else if (m.includes('download'))  label = this.t('downloading_video');
    else if (m.includes('pars'))      label = this.t('parsing_video');
    else if (m.includes('transcrib')) label = this.t('transcribing_audio');
    else if (m.includes('optimiz'))   label = this.t('optimizing_transcript');
    else if (m.includes('summary'))   label = this.t('generating_summary');
    else if (m.includes('complet'))   label = this.t('completed');
    else if (m.includes('prepar'))    label = this.t('preparing');

    this.progressMessage.textContent = label;
  }

  _showProgress() {
    this.emptyState.style.display    = 'none';
    this.resultsPanel.classList.remove('show');
    this.progressPanel.classList.add('show');
    // Reset mode badge & progress bar color for new task
    if (this.modeBadge) { this.modeBadge.style.display = 'none'; this.modeBadge.className = 'mode-badge'; }
    if (this.progressFill) this.progressFill.classList.remove('subtitle-mode');
  }
  _hideProgress() { this.progressPanel.classList.remove('show'); }

  /* ── Results ──────────────────────────────────────────── */
  /** Aligns with backend Translator.normalize_lang_code for tab visibility. */
  _normLangTab(code) {
    if (!code) return '';
    const c = String(code).toLowerCase().trim();
    if (c.startsWith('zh')) return 'zh';
    if (c.length >= 2) return c.slice(0, 2);
    return c;
  }

  _showResults(script, summary, videoTitle, translation, detectedLang, summaryLang, statistics) {
    this.scriptContent.innerHTML  = script    ? marked.parse(script)      : '';
    this.summaryContent.innerHTML = summary   ? marked.parse(summary)     : '';
    this.currentStatistics = statistics || null;
    this._renderStatistics(this.currentStatistics);

    const d = this._normLangTab(detectedLang);
    const s = this._normLangTab(summaryLang);
    const showTranslation = Boolean(translation) && d && s && d !== s;
    if (showTranslation) {
      this.translationContent.innerHTML = marked.parse(translation);
      this.translationTabBtn.style.display  = 'inline-block';
      this.dlTranslation.style.display      = 'inline-flex';
    } else {
      this.translationContent.innerHTML = '';
      this.translationTabBtn.style.display  = 'none';
      this.dlTranslation.style.display      = 'none';
    }

    this.resultsPanel.classList.add('show');
    this._switchTab('summary');
    this.resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  _renderStatistics(statistics) {
    if (!this.statisticsContent) return;
    if (!statistics || typeof statistics !== 'object') {
      this.statisticsContent.innerHTML = `<div class="stats-empty">${this._escapeHtml(this.t('statistics_unavailable'))}</div>`;
      return;
    }

    const rows = [
      [this.t('stat_processing_time'), this._formatDuration(statistics.processing_seconds, statistics.processing_minutes)],
      [this.t('stat_input'), statistics.input_name || statistics.source_ref || '—'],
      [this.t('stat_source_type'), this._formatSourceType(statistics.input_type)],
      [this.t('stat_extraction'), this._formatExtraction(statistics.extraction_method)],
      [this.t('stat_started'), this._formatDate(statistics.processing_started_at)],
      [this.t('stat_finished'), this._formatDate(statistics.processing_finished_at)],
      [this.t('stat_detected_language'), statistics.detected_language || '—'],
      [this.t('stat_summary_language'), statistics.summary_language || '—'],
      [this.t('stat_translation'), statistics.translation_generated ? this.t('stat_yes') : this.t('stat_no')],
      [this.t('stat_model'), statistics.model || '—'],
      [this.t('stat_raw_transcript'), this._formatTextUnits(statistics.raw_transcript)],
      [this.t('stat_optimized_transcript'), this._formatTextUnits(statistics.optimized_transcript)],
      [this.t('stat_summary'), this._formatTextUnits(statistics.summary)],
    ];

    if (statistics.translation_generated) {
      rows.push([this.t('translation'), this._formatTextUnits(statistics.translation)]);
    }

    this.statisticsContent.innerHTML = `
      <div class="stats-list">
        ${rows.map(([label, value]) => `
          <div class="stat-row">
            <div class="stat-label">${this._escapeHtml(label)}</div>
            <div class="stat-value">${this._escapeHtml(value)}</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  _formatDuration(seconds, minutes) {
    const sec = Number(seconds);
    const min = Number(minutes);
    if (!Number.isFinite(sec)) return '—';
    const mins = Number.isFinite(min) ? min : sec / 60;
    return `${mins.toFixed(2)} ${this.t('stat_minutes')} (${Math.round(sec)} ${this.t('stat_seconds')})`;
  }

  _formatDate(value) {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString();
  }

  _formatTextUnits(units) {
    if (!units || typeof units !== 'object') return '—';
    const chars = Number(units.chars) || 0;
    const words = Number(units.words) || 0;
    const lines = Number(units.lines) || 0;
    return `${chars} ${this.t('stat_chars')} · ${words} ${this.t('stat_words')} · ${lines} ${this.t('stat_lines')}`;
  }

  _formatSourceType(value) {
    const v = String(value || '').toLowerCase();
    if (v === 'url') return 'URL';
    if (v === 'upload') return 'Upload';
    return value || '—';
  }

  _formatExtraction(value) {
    const v = String(value || '').toLowerCase();
    const en = {
      subtitle: 'Native subtitles',
      whisper: 'Whisper transcription',
      text_upload: 'Text file',
      whisper_upload: 'Uploaded media + Whisper',
    };
    return en[v] || value || '—';
  }

  _escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  _hideResults() { this.resultsPanel.classList.remove('show'); }

  /* ── Tabs ─────────────────────────────────────────────── */
  _switchTab(name) {
    this.tabBtns.forEach(b  => b.classList.toggle('active',  b.dataset.tab === name));
    this.tabPanes.forEach(p => p.classList.toggle('active', p.id === `${name}Tab`));
  }

  /* ── Download ─────────────────────────────────────────── */
  async _downloadFile(type) {
    if (!this.currentTaskId) { this._showError(this.t('error_no_download')); return; }
    try {
      const r = await fetch(`${this.apiBase}/task-status/${this.currentTaskId}`);
      if (!r.ok) throw new Error('Failed to get task status');
      const task = await r.json();

      let filename;
      if      (type === 'script')      filename = task.script_filename      || (task.script_path      ? task.script_path.split('/').pop()      : `transcript_${task.safe_title||'x'}_${task.short_id||'x'}.md`);
      else if (type === 'summary')     filename = task.summary_filename     || (task.summary_path     ? task.summary_path.split('/').pop()     : `summary_${task.safe_title||'x'}_${task.short_id||'x'}.md`);
      else if (type === 'translation') filename = task.translation_filename || (task.translation_path ? task.translation_path.split('/').pop() : `translation_${task.safe_title||'x'}_${task.short_id||'x'}.md`);
      else throw new Error('Unknown type');

      const a = document.createElement('a');
      a.href = `${this.apiBase}/download/${encodeURIComponent(filename)}`;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (e) {
      this._showError(this.t('error_download_failed') + e.message);
    }
  }

  /* ── UI helpers ───────────────────────────────────────── */
  _setLoading(on) {
    this.submitBtn.disabled = on;
    this.submitBtn.innerHTML = on
      ? `<span class="spinner"></span> ${this.t('processing')}`
      : `<i class="fas fa-search"></i> <span>${this.t('start_transcription')}</span>`;
    if (this.uploadPickBtn) this.uploadPickBtn.disabled = on;
    if (this.uploadZone) {
      this.uploadZone.style.pointerEvents = on ? 'none' : '';
      this.uploadZone.style.opacity = on ? '0.65' : '';
      this.uploadZone.tabIndex = on ? -1 : 0;
    }
    if (this.fileInput) this.fileInput.disabled = on;
  }

  _showError(msg) {
    this.errorMsg.textContent = msg;
    this.errorBanner.classList.add('show');
    this.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    setTimeout(() => this._hideError(), 6000);
  }
  _hideError() { this.errorBanner.classList.remove('show'); }

  _debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }
}

/* ── Boot ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  window.vt = new VideoTranscriber();
});

window.addEventListener('beforeunload', () => {
  if (window.vt?.eventSource) window.vt._stopSSE();
});
