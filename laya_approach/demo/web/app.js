/* Onda Travel browser client.
 *
 * Binary websocket frames are 24 kHz float32 PCM in both directions.  JSON
 * frames carry the conversation, Laya's decisions, the catalog and playback
 * control.  The mic stays open while Laya speaks; the stop button and a user
 * barge-in both cut scheduled audio immediately.
 */
(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  // ---------------------------------------------------------------- copy
  const I18N = {
    en: {
      'a11y.skip': 'Skip to trips',
      'nav.collection': 'The collection',
      'nav.how': 'How it works',
      'nav.live': 'Live voice demo',
      'hero.eyebrow': 'Voice-first travel design',
      'hero.title1': 'Say it.',
      'hero.title2': 'See it.',
      'hero.title3': 'Go.',
      'hero.sub': 'Talk to Laya like a person. She listens while you speak, reshapes the collection in real time, and answers in English or Portuguese — whichever language you choose.',
      'hero.meta1': 'Full-duplex turn-taking',
      'hero.meta2': 'English · Português',
      'hero.meta3': '12 mock escapes',
      'hero.cta': 'Start a voice session',
      'hero.ctaNote': 'or type below — no account needed',
      'hero.postcardKicker': 'Live itinerary',
      'hero.postcardTitle': '“Somewhere warm, with a good beach and food.”',
      'hero.note': 'Laya is taking the floor',
      'agent.kicker': 'Your travel designer',
      'agent.empty': 'Tell me what the trip should feel like. I’ll do the filtering.',
      'agent.pause': 'Pause mic',
      'agent.resume': 'Start mic',
      'agent.reset': 'Start over',
      'agent.telemetry': 'Laya telemetry',
      'agent.greeting': 'Hi! I’m Laya, your travel designer. Tell me what kind of trip you’re imagining, and I’ll shape the catalog while we talk.',
      'voice.start': 'Tap to speak',
      'voice.listening': 'Listening — tap to pause',
      'voice.thinking': 'Thinking…',
      'voice.speaking': 'Speaking — tap to interrupt',
      'voice.connecting': 'Waking up…',
      'voice.off': 'Mic is off',
      'state.idle': 'Ready',
      'state.listening': 'Listening',
      'state.thinking': 'Thinking',
      'state.speaking': 'Speaking',
      'state.connecting': 'Connecting',
      'state.error': 'Offline',
      'composer.placeholder': 'Or type a message…',
      'results.eyebrow': 'The collection',
      'results.title': 'Trips with a point of view',
      'results.clear': 'Clear filters',
      'results.note': 'Mock inventory for the demo · prices are per person, in USD',
      'results.noFilters': 'No filters yet — just talk, or use the mic.',
      'empty.title': 'Nothing matches all of that — yet.',
      'empty.body': 'Try relaxing the budget or the number of nights. Laya can help you choose what to let go of.',
      'empty.ask': 'Ask Laya for ideas',
      'empty.clear': 'Show everything',
      'how.eyebrow': 'How it works',
      'how.title': 'A conversation, not a search box.',
      'how.step1.title': 'You talk naturally',
      'how.step1.body': 'Pause, think out loud, change your mind. The mic stays open while Laya speaks, so you can interrupt her mid-sentence.',
      'how.step2.title': 'Laya takes the floor',
      'how.step2.body': 'The Laya companion reads the ASR stream and decides when your turn is really over — punctuation alone never fools it.',
      'how.step3.title': 'The page follows',
      'how.step3.body': 'Qwen 3 4B turns your intent into filters, Phonon speaks the answer, and the collection reshuffles while the sentence is still landing.',
      'footer.note': 'A reference demo for the Laya duplex companion. All trips and prices are fictional.',
      'boot.loading': 'Waking Laya and the travel desk…',
      'boot.ready': 'Connected. Tap the orb to talk.',
      'telemetry.title': 'Companion telemetry',
      'telemetry.tag': 'Last decision',
      'telemetry.latency': 'Latency',
      'telemetry.intent': 'Turn intent',
      'telemetry.noPolicy': 'heuristic fallback',
      'card.view': 'View trip',
      'card.from': 'from',
      'card.perPerson': 'per person',
      'card.nights': 'nights',
      'card.night': 'night',
      'card.topPick': 'Top pick',
      'card.match': 'Match',
      'card.rating': 'rating',
      'modal.close': 'Close',
      'modal.highlights': 'Highlights',
      'modal.bestFor': 'Best for',
      'modal.ask': 'Ask Laya about this',
      'modal.priceNote': 'per person for the whole stay',
      'modal.saved': 'Saved to your shortlist',
      'toast.micDenied': 'Microphone permission was denied. You can still type to Laya.',
      'toast.micUnsupported': 'This browser does not expose microphone capture. Use the text box or Chrome/Edge over localhost.',
      'toast.connectionLost': 'Connection lost. Reconnecting…',
      'toast.reconnected': 'Reconnected.',
      'toast.language': 'Interface switched to English.',
      'toast.notReady': 'Laya is still warming up. One moment.',
      'toast.error': 'Something went wrong. Try again, or reset the session.',
      'toast.audio': 'Tap the page once to enable audio playback.',
      'toast.empty': 'Try a looser brief, or clear a filter.',
      'toast.micOn': 'Mic is live. Speak naturally — Laya will wait for your turn.',
      'toast.micOff': 'Mic paused.',
      'toast.noMicAudio': 'The mic is open but no audio is reaching the page. Check the input device or hard-refresh the tab.',
      'welcome.play': 'Play Laya’s welcome',
      'degraded.mic': 'microphone',
      'degraded.tts': 'voice output',
      'degraded.llm': 'Qwen replies',
      'degraded.policy': 'Laya turn-taking',
      'degraded.banner': 'Degraded mode: {items} unavailable. The page still works with text and the deterministic catalog. Check /api/health.',
      'tag.user_finish_speaking': 'turn complete',
      'tag.user_is_speaking': 'user speaking',
      'tag.user_interruption': 'barge-in',
      'tag.user_backchannel': 'backchannel',
      'tag.user_is_thinking': 'thinking',
      'tag.system_backchannel': 'listener signal',
      'tag.no_tag': 'no action',
      'meta.you': 'You',
      'meta.laya': 'Laya',
      'meta.voice': 'voice',
      'meta.text': 'typed',
      'meta.listening': 'listening…',
      'suggestions.1': 'I want a beach with good food under $1,500',
      'suggestions.2': 'A romantic mountain escape for two',
      'suggestions.3': 'Somewhere warm for the family',
      'suggestions.4': 'Switch to Portuguese',
      'suggestions.5': 'Show me everything again',
      'suggestions.6': 'Tell me a joke about the beach',
    },
    'pt-BR': {
      'a11y.skip': 'Ir para as viagens',
      'nav.collection': 'A coleção',
      'nav.how': 'Como funciona',
      'nav.live': 'Demo de voz ao vivo',
      'hero.eyebrow': 'Viagem desenhada por voz',
      'hero.title1': 'Fale.',
      'hero.title2': 'Veja.',
      'hero.title3': 'Vá.',
      'hero.sub': 'Converse com a Laya como com uma pessoa. Ela escuta enquanto você fala, refaz a coleção em tempo real e responde em português ou inglês — no idioma que você escolher.',
      'hero.meta1': 'Turnos em full-duplex',
      'hero.meta2': 'Português · English',
      'hero.meta3': '12 escapadas fictícias',
      'hero.cta': 'Começar uma conversa por voz',
      'hero.ctaNote': 'ou digite abaixo — sem criar conta',
      'hero.postcardKicker': 'Roteiro ao vivo',
      'hero.postcardTitle': '“Um lugar quente, com boa praia e boa comida.”',
      'hero.note': 'Laya está assumindo o turno',
      'agent.kicker': 'Sua consultora de viagens',
      'agent.empty': 'Me conte como a viagem deveria ser. Eu cuido dos filtros.',
      'agent.pause': 'Pausar microfone',
      'agent.resume': 'Ligar microfone',
      'agent.reset': 'Começar de novo',
      'agent.telemetry': 'Telemetria da Laya',
      'agent.greeting': 'Oi! Eu sou a Laya, sua consultora de viagens. Me conte que tipo de viagem você imagina, e eu ajusto o catálogo enquanto conversamos.',
      'voice.start': 'Toque para falar',
      'voice.listening': 'Ouvindo — toque para pausar',
      'voice.thinking': 'Pensando…',
      'voice.speaking': 'Falando — toque para interromper',
      'voice.connecting': 'Acordando…',
      'voice.off': 'Microfone desligado',
      'state.idle': 'Pronta',
      'state.listening': 'Ouvindo',
      'state.thinking': 'Pensando',
      'state.speaking': 'Falando',
      'state.connecting': 'Conectando',
      'state.error': 'Offline',
      'composer.placeholder': 'Ou digite uma mensagem…',
      'results.eyebrow': 'A coleção',
      'results.title': 'Viagens com um ponto de vista',
      'results.clear': 'Limpar filtros',
      'results.note': 'Inventário fictício para a demonstração · preços por pessoa, em dólares',
      'results.noFilters': 'Nenhum filtro ainda — é só falar, ou usar o microfone.',
      'empty.title': 'Nada combina com tudo isso — ainda.',
      'empty.body': 'Tente relaxar o orçamento ou o número de noites. A Laya pode ajudar a escolher o que soltar.',
      'empty.ask': 'Pedir ideias à Laya',
      'empty.clear': 'Mostrar tudo',
      'how.eyebrow': 'Como funciona',
      'how.title': 'Uma conversa, não uma caixa de busca.',
      'how.step1.title': 'Você fala naturalmente',
      'how.step1.body': 'Pause, pense em voz alta, mude de ideia. O microfone continua aberto enquanto a Laya fala, então você pode interromper no meio da frase.',
      'how.step2.title': 'A Laya assume o turno',
      'how.step2.body': 'A companion Laya lê o fluxo do ASR e decide quando o seu turno realmente terminou — pontuação sozinha nunca a engana.',
      'how.step3.title': 'A página acompanha',
      'how.step3.body': 'O Qwen 3 4B transforma sua intenção em filtros, a Phonon fala a resposta e a coleção se reorganiza enquanto a frase ainda está no ar.',
      'footer.note': 'Uma demo de referência para a companion duplex Laya. Todas as viagens e preços são fictícios.',
      'boot.loading': 'Acordando a Laya e a mesa de viagens…',
      'boot.ready': 'Conectado. Toque na esfera para falar.',
      'telemetry.title': 'Telemetria da companion',
      'telemetry.tag': 'Última decisão',
      'telemetry.latency': 'Latência',
      'telemetry.intent': 'Intenção do turno',
      'telemetry.noPolicy': 'fallback heurístico',
      'card.view': 'Ver viagem',
      'card.from': 'a partir de',
      'card.perPerson': 'por pessoa',
      'card.nights': 'noites',
      'card.night': 'noite',
      'card.topPick': 'Escolha da Laya',
      'card.match': 'Combina',
      'card.rating': 'avaliação',
      'modal.close': 'Fechar',
      'modal.highlights': 'Destaques',
      'modal.bestFor': 'Ideal para',
      'modal.ask': 'Perguntar à Laya sobre esta viagem',
      'modal.priceNote': 'por pessoa, pela estadia completa',
      'modal.saved': 'Salvo na sua lista',
      'toast.micDenied': 'A permissão do microfone foi negada. Você ainda pode digitar para a Laya.',
      'toast.micUnsupported': 'Este navegador não expõe captura de microfone. Use a caixa de texto ou Chrome/Edge em localhost.',
      'toast.connectionLost': 'Conexão perdida. Reconectando…',
      'toast.reconnected': 'Reconectado.',
      'toast.language': 'Interface alterada para português.',
      'toast.notReady': 'A Laya ainda está acordando. Um instante.',
      'toast.error': 'Algo deu errado. Tente de novo ou reinicie a sessão.',
      'toast.audio': 'Toque na página uma vez para liberar o áudio.',
      'toast.empty': 'Tente um pedido mais aberto, ou limpe um filtro.',
      'toast.micOn': 'Microfone aberto. Fale naturalmente — a Laya espera o seu turno.',
      'toast.micOff': 'Microfone pausado.',
      'toast.noMicAudio': 'O microfone está aberto, mas nenhum áudio chega à página. Verifique o dispositivo de entrada ou recarregue a aba.',
      'welcome.play': 'Ouvir a saudação da Laya',
      'degraded.mic': 'microfone',
      'degraded.tts': 'saída de voz',
      'degraded.llm': 'respostas do Qwen',
      'degraded.policy': 'turn-taking da Laya',
      'degraded.banner': 'Modo degradado: {items} indisponível(is). A página continua funcionando com texto e o catálogo determinístico. Veja /api/health.',
      'tag.user_finish_speaking': 'turno concluído',
      'tag.user_is_speaking': 'usuário falando',
      'tag.user_interruption': 'interrupção',
      'tag.user_backchannel': 'backchannel',
      'tag.user_is_thinking': 'pensando',
      'tag.system_backchannel': 'sinal de escuta',
      'tag.no_tag': 'nenhuma ação',
      'meta.you': 'Você',
      'meta.laya': 'Laya',
      'meta.voice': 'voz',
      'meta.text': 'digitado',
      'meta.listening': 'ouvindo…',
      'suggestions.1': 'Quero uma praia com boa comida até US$ 1.500',
      'suggestions.2': 'Uma viagem romântica na montanha para dois',
      'suggestions.3': 'Um lugar quente para a família',
      'suggestions.4': 'Trocar para inglês',
      'suggestions.5': 'Mostrar tudo de novo',
      'suggestions.6': 'Me conte uma piada sobre praia',
    },
  };

  const TAG_LABELS = {
    en: {
      beach: 'Beach', islands: 'Islands', mountains: 'Mountains', city: 'City',
      culture: 'Culture', food: 'Food', nightlife: 'Nightlife', nature: 'Nature',
      family: 'Family', romantic: 'Romantic', adventure: 'Adventure',
      budget: 'Budget', luxury: 'Luxury', history: 'History', wildlife: 'Wildlife',
      diving: 'Diving', hiking: 'Hiking', skiing: 'Skiing', wine: 'Wine',
      theme_parks: 'Theme parks', shopping: 'Shopping', music: 'Music',
    },
    'pt-BR': {
      beach: 'Praia', islands: 'Ilhas', mountains: 'Montanhas', city: 'Cidade',
      culture: 'Cultura', food: 'Gastronomia', nightlife: 'Vida noturna', nature: 'Natureza',
      family: 'Família', romantic: 'Romântico', adventure: 'Aventura',
      budget: 'Econômico', luxury: 'Luxo', history: 'História', wildlife: 'Vida selvagem',
      diving: 'Mergulho', hiking: 'Trilhas', skiing: 'Esqui', wine: 'Vinhos',
      theme_parks: 'Parques temáticos', shopping: 'Compras', music: 'Música',
    },
  };
  const REGION_LABELS = {
    en: { south_america: 'South America', north_america: 'North America', europe: 'Europe', asia: 'Asia', africa: 'Africa' },
    'pt-BR': { south_america: 'América do Sul', north_america: 'América do Norte', europe: 'Europa', asia: 'Ásia', africa: 'África' },
  };
  const CLIMATE_LABELS = {
    en: { warm: 'Warm', mild: 'Mild', cold: 'Cold' },
    'pt-BR': { warm: 'Quente', mild: 'Ameno', cold: 'Frio' },
  };
  const PACE_LABELS = {
    en: { relaxed: 'Relaxed', balanced: 'Balanced', adventure: 'Adventure' },
    'pt-BR': { relaxed: 'Tranquilo', balanced: 'Equilibrado', adventure: 'Aventura' },
  };

  const SUGGESTIONS = {
    en: ['suggestions.1', 'suggestions.2', 'suggestions.3', 'suggestions.4', 'suggestions.5', 'suggestions.6'],
    'pt-BR': ['suggestions.1', 'suggestions.2', 'suggestions.3', 'suggestions.4', 'suggestions.5', 'suggestions.6'],
  };

  function loadFavorites() {
    try {
      const parsed = JSON.parse(localStorage.getItem('onda:favorites') || '[]');
      return new Set(Array.isArray(parsed) ? parsed : []);
    } catch (error) {
      return new Set();
    }
  }

  function saveFavorites() {
    try {
      localStorage.setItem('onda:favorites', JSON.stringify([...state.favorites]));
    } catch (error) { /* private mode */ }
  }

  const prefersPt = (navigator.language || '').toLowerCase().startsWith('pt');
  const state = {
    language: prefersPt ? 'pt-BR' : 'en',
    ws: null,
    connected: false,
    ready: false,
    micOn: false,
    micCtx: null,
    micStream: null,
    micNode: null,
    micSource: null,
    micFrameCount: 0,
    micFrameTimer: null,
    welcomePlayed: false,
    welcomeRequested: false,
    welcomeTimer: null,
    suppressMic: false,
    audioKind: '',
    audioCtx: null,
    nextPlayTime: 0,
    sources: [],
    pendingAudioEnd: null,
    audioSeq: null,
    audioEndTimer: null,
    items: [],
    filters: { tags: [], max_price: null, min_price: null, min_nights: null, max_nights: null, region: null, climate: null, pace: null },
    highlights: [],
    total: 0,
    messages: [],
    pendingUser: '',
    pendingAssistant: null,
    status: 'connecting',
    capabilities: null,
    serverWarnings: [],
    lastPolicy: null,
    lastTurnForced: false,
    lastIntent: '',
    reconnectTimer: null,
    reconnectAttempt: 0,
    bootHidden: false,
    favorites: loadFavorites(),
    toastTimer: null,
    level: 0,
    waveBars: [],
    lastWaveUpdate: 0,
    modalItem: null,
  };

  // ---------------------------------------------------------------- utils
  const t = (key) => (I18N[state.language] && I18N[state.language][key]) || (I18N.en[key] || key);
  const tagLabel = (tag) => (TAG_LABELS[state.language] && TAG_LABELS[state.language][tag]) || tag;
  const regionLabel = (value) => (REGION_LABELS[state.language] && REGION_LABELS[state.language][value]) || value;
  const climateLabel = (value) => (CLIMATE_LABELS[state.language] && CLIMATE_LABELS[state.language][value]) || value;
  const paceLabel = (value) => (PACE_LABELS[state.language] && PACE_LABELS[state.language][value]) || value;
  const locale = () => (state.language === 'pt-BR' ? 'pt-BR' : 'en-US');
  const money = (value) => new Intl.NumberFormat(locale(), { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);
  const nightsLabel = (n) => `${n} ${n === 1 ? t('card.night') : t('card.nights')}`;
  const statusLabel = (status) => t(`state.${status}`) || t('state.idle');

  function esc(value) {
    return String(value == null ? '' : value)
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
  }

  function toast(message, error = false) {
    const el = $('#toast');
    if (!el) return;
    el.textContent = message;
    el.classList.toggle('is-error', Boolean(error));
    el.classList.add('is-visible');
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => el.classList.remove('is-visible'), 4200);
  }

  function hideBoot() {
    if (state.bootHidden) return;
    state.bootHidden = true;
    const pill = $('#bootPill');
    if (pill) {
      pill.classList.add('is-hidden');
      setTimeout(() => { pill.style.display = 'none'; }, 400);
    }
  }

  function bootError(message) {
    const pill = $('#bootPill');
    if (!pill) return;
    pill.classList.remove('is-hidden');
    pill.classList.add('is-error');
    pill.querySelector('.boot-spinner')?.remove();
    $('#bootText').textContent = message;
    state.bootHidden = false;
  }

  // --------------------------------------------------------------- language
  function applyLanguage(lang, { notifyServer = false, showToast = false } = {}) {
    if (!I18N[lang]) lang = 'en';
    state.language = lang;
    document.documentElement.lang = lang;
    document.documentElement.dataset.language = lang;

    $$('[data-i18n]').forEach((el) => {
      const key = el.getAttribute('data-i18n');
      const value = I18N[lang] && I18N[lang][key];
      if (value) el.textContent = value;
    });
    $$('[data-i18n-placeholder]').forEach((el) => {
      const key = el.getAttribute('data-i18n-placeholder');
      const value = I18N[lang] && I18N[lang][key];
      if (value) el.setAttribute('placeholder', value);
    });
    $$('.lang-btn').forEach((btn) => {
      const active = btn.dataset.lang === lang;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });

    for (const message of state.messages) {
      if (message.meta?.greeting) message.text = t('agent.greeting');
    }
    renderSuggestions();
    renderFilters();
    renderMessages();
    if (state.ready || state.items.length) renderCards();
    updateDegradedBanner();
    setStatus(state.status);
    if (state.modalItem) openModal(state.modalItem.id, { focus: false });
    if (notifyServer && state.ws && state.ws.readyState === WebSocket.OPEN && state.ready) {
      send({ type: 'language', language: lang });
    }
    if (showToast) toast(t('toast.language'));
  }

  // -------------------------------------------------------------- websocket
  function connect() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) return;
    if (state.reconnectTimer) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    let ws;
    try {
      ws = new WebSocket(`${protocol}//${location.host}/ws`);
    } catch (error) {
      setStatus('error');
      bootError(t('toast.error'));
      return;
    }
    ws.binaryType = 'arraybuffer';
    state.ws = ws;
    setStatus('connecting');

    ws.onopen = () => {
      state.connected = true;
      state.reconnectAttempt = 0;
      if (state.bootHidden) {
        // Reconnect: do not show the whole boot pill again.
        hideBoot();
        toast(t('toast.reconnected'));
      }
      ws.send(JSON.stringify({ type: 'ping', t: Date.now() }));
    };
    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try { handleMessage(JSON.parse(event.data)); } catch (error) { console.warn('bad message', error); }
      } else if (event.data instanceof ArrayBuffer) {
        playPcm(event.data);
      }
    };
    ws.onclose = () => {
      state.connected = false;
      state.ready = false;
      state.ws = null;
      stopMic({ silent: true, send: false });
      stopAudio();
      setStatus('error');
      if (state.reconnectAttempt < 12) {
        const delay = Math.min(8000, 900 + state.reconnectAttempt * 900);
        state.reconnectAttempt += 1;
        toast(t('toast.connectionLost'), true);
        state.reconnectTimer = setTimeout(connect, delay);
      } else {
        bootError(t('toast.connectionLost'));
      }
    };
    ws.onerror = () => { /* onclose handles the retry */ };
  }

  function send(payload) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return false;
    state.ws.send(JSON.stringify(payload));
    return true;
  }

  // --------------------------------------------------------------- messages
  function addMessage(role, text, meta = {}) {
    if (!text) return;
    state.messages.push({ role, text, meta });
    if (state.messages.length > 50) state.messages.shift();
  }

  function commitAssistant() {
    if (!state.pendingAssistant) return;
    const pending = state.pendingAssistant;
    if (pending.text) {
      addMessage('assistant', pending.text, { interrupted: pending.interrupted });
    }
    state.pendingAssistant = null;
  }

  function renderMessages() {
    const box = $('#transcript');
    if (!box) return;
    box.innerHTML = '';
    if (!state.messages.length && !state.pendingUser && !state.pendingAssistant) {
      const empty = document.createElement('div');
      empty.className = 'chat-empty';
      empty.id = 'chatEmpty';
      empty.innerHTML = `<p class="chat-empty-mark">✦</p><p>${esc(t('agent.empty'))}</p>`;
      box.appendChild(empty);
      return;
    }

    for (const message of state.messages) {
      const row = document.createElement('div');
      row.className = `msg msg-${message.role}`;
      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';
      if (message.meta?.interrupted) bubble.classList.add('is-interrupted');
      bubble.textContent = message.text;
      if (message.meta?.greeting) {
        const play = document.createElement('button');
        play.type = 'button';
        play.className = 'welcome-play';
        play.title = t('welcome.play');
        play.setAttribute('aria-label', t('welcome.play'));
        play.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9v6h4l5 4V5L8 9zm12.5 3a4.5 4.5 0 0 0-2-3.7v7.4a4.5 4.5 0 0 0 2-3.7z" fill="currentColor"/></svg>';
        play.addEventListener('click', (event) => {
          event.stopPropagation();
          requestWelcome(true);
        });
        bubble.append(' ', play);
      }
      const meta = document.createElement('div');
      meta.className = 'msg-meta';
      if (message.role === 'assistant') {
        meta.textContent = t('meta.laya');
      } else {
        meta.textContent = t('meta.you');
      }
      row.append(bubble, meta);
      box.appendChild(row);
    }

    if (state.pendingUser) {
      const row = document.createElement('div');
      row.className = 'msg msg-user msg-pending';
      row.innerHTML = `<div class="msg-bubble">${esc(state.pendingUser)}</div>`;
      box.appendChild(row);
    }

    if (state.pendingAssistant) {
      const row = document.createElement('div');
      row.className = 'msg msg-assistant';
      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';
      if (state.pendingAssistant.interrupted) bubble.classList.add('is-interrupted');
      if (state.pendingAssistant.typing) {
        bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
      } else {
        bubble.textContent = state.pendingAssistant.text || '';
      }
      const meta = document.createElement('div');
      meta.className = 'msg-meta';
      meta.textContent = t('meta.laya');
      row.append(bubble, meta);
      box.appendChild(row);
    }
    box.scrollTop = box.scrollHeight;
  }

  function renderSuggestions() {
    const box = $('#suggestions');
    if (!box) return;
    box.innerHTML = '';
    for (const key of SUGGESTIONS[state.language] || []) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'suggestion';
      button.textContent = t(key);
      button.addEventListener('click', () => sendText(t(key)));
      box.appendChild(button);
    }
  }

  function updateDegradedBanner() {
    const banner = $('#degradedBanner');
    if (!banner) return;
    if (!state.ready || !state.serverWarnings.length) {
      banner.hidden = true;
      banner.textContent = '';
      return;
    }
    const caps = state.capabilities || {};
    const missing = [];
    if (!caps.stt) missing.push(t('degraded.mic'));
    if (!caps.tts) missing.push(t('degraded.tts'));
    if (!caps.llm) missing.push(t('degraded.llm'));
    if (!caps.policy) missing.push(t('degraded.policy'));
    if (!missing.length) {
      banner.hidden = true;
      banner.textContent = '';
      return;
    }
    banner.hidden = false;
    banner.textContent = t('degraded.banner').replace('{items}', missing.join(', '));
    if (state.serverWarnings.length) banner.title = state.serverWarnings.join('\n');
  }

  // ---------------------------------------------------------------- status
  function setStatus(status) {
    state.status = status;
    const panel = $('#agentPanel');
    const chip = $('#statusChip');
    if (panel) {
      panel.classList.remove('is-listening', 'is-thinking', 'is-speaking');
      if (status === 'listening') panel.classList.add('is-listening');
      if (status === 'thinking') panel.classList.add('is-thinking');
      if (status === 'speaking') panel.classList.add('is-speaking');
    }
    if (chip) {
      chip.classList.remove('is-listening', 'is-thinking', 'is-speaking', 'is-error');
      if (['listening', 'thinking', 'speaking'].includes(status)) chip.classList.add(`is-${status}`);
      if (status === 'error') chip.classList.add('is-error');
    }
    const label = $('#statusText');
    if (label) label.textContent = statusLabel(status);
    const voice = $('#voiceLabel');
    if (voice) {
      if (status === 'listening') voice.textContent = t('voice.listening');
      else if (status === 'thinking') voice.textContent = t('voice.thinking');
      else if (status === 'speaking') voice.textContent = t('voice.speaking');
      else if (status === 'connecting') voice.textContent = t('voice.connecting');
      else if (status === 'error') voice.textContent = t('state.error');
      else voice.textContent = t('voice.start');
    }
    const mic = $('#micBtn');
    if (mic) mic.setAttribute('aria-pressed', state.micOn ? 'true' : 'false');
    const pause = $('#pauseBtn');
    if (pause) {
      const text = pause.querySelector('span:last-child');
      if (text) text.textContent = state.micOn ? t('agent.pause') : t('agent.resume');
      pause.classList.toggle('is-off', !state.micOn);
    }
  }

  // --------------------------------------------------------------- catalog
  function formatFilterLabel(key, value) {
    if (key === 'tags') return tagLabel(value);
    if (key === 'max_price') return state.language === 'pt-BR' ? `Até ${money(value)}` : `Under ${money(value)}`;
    if (key === 'min_price') return state.language === 'pt-BR' ? `Acima de ${money(value)}` : `From ${money(value)}`;
    if (key === 'nights') return nightsLabel(value);
    if (key === 'min_nights') return state.language === 'pt-BR' ? `${value}+ noites` : `${value}+ nights`;
    if (key === 'max_nights') return state.language === 'pt-BR' ? `Até ${value} noites` : `Up to ${value} nights`;
    if (key === 'region') return regionLabel(value);
    if (key === 'climate') return climateLabel(value);
    if (key === 'pace') return paceLabel(value);
    return String(value);
  }

  function renderFilters() {
    const bar = $('#filterBar');
    const clear = $('#clearFilters');
    if (!bar) return;
    bar.innerHTML = '';
    const chips = [];
    const filters = state.filters || {};
    (filters.tags || []).forEach((tag) => chips.push({ key: 'tags', value: tag, soft: true }));
    const exactNights = filters.min_nights && filters.max_nights && filters.min_nights === filters.max_nights;
    if (exactNights) chips.push({ key: 'nights', value: filters.min_nights, soft: false });
    for (const key of ['max_price', 'min_price', 'min_nights', 'max_nights', 'region', 'climate', 'pace']) {
      if (exactNights && (key === 'min_nights' || key === 'max_nights')) continue;
      if (filters[key] !== null && filters[key] !== undefined && filters[key] !== '') {
        chips.push({ key, value: filters[key], soft: false });
      }
    }
    if (!chips.length) {
      const hint = document.createElement('span');
      hint.className = 'filter-empty-hint';
      hint.textContent = t('results.noFilters');
      bar.appendChild(hint);
      if (clear) clear.hidden = true;
      return;
    }
    if (clear) clear.hidden = false;
    chips.forEach((chip) => {
      const element = document.createElement('span');
      element.className = `filter-chip${chip.soft ? ' is-soft' : ''}`;
      const label = document.createElement('span');
      label.textContent = formatFilterLabel(chip.key, chip.value);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.setAttribute('aria-label', `${t('results.clear')}: ${label.textContent}`);
      remove.textContent = '×';
      remove.addEventListener('click', () => removeFilter(chip.key, chip.value));
      element.append(label, remove);
      bar.appendChild(element);
    });
  }

  function removeFilter(key, value) {
    const next = {
      ...state.filters,
      tags: [...(state.filters.tags || [])],
    };
    if (key === 'tags') next.tags = next.tags.filter((tag) => tag !== value);
    else if (key === 'nights') { next.min_nights = null; next.max_nights = null; }
    else next[key] = null;
    state.filters = next;
    renderFilters();
    send({ type: 'filters', filters: next });
  }

  function clearFilters() {
    state.filters = { tags: [], max_price: null, min_price: null, min_nights: null, max_nights: null, region: null, climate: null, pace: null };
    renderFilters();
    send({ type: 'filters', filters: state.filters });
  }

  function sceneSvg(item) {
    const c1 = item.colors?.[0] || '#9adfd4';
    const c2 = item.colors?.[1] || '#0e6e63';
    const id = String(item.id || 'trip').replace(/[^a-z0-9]/gi, '');
    const sky = `sky-${id}`;
    const sea = `sea-${id}`;
    const defs = `<defs>
      <linearGradient id="${sky}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="${c1}"/><stop offset="1" stop-color="${c2}"/>
      </linearGradient>
      <linearGradient id="${sea}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#f7fbf8" stop-opacity=".62"/><stop offset="1" stop-color="${c2}" stop-opacity=".94"/>
      </linearGradient>
    </defs>`;
    const sun = `<circle cx="315" cy="65" r="34" fill="#fff4d6" opacity=".82"/>`;
    const base = `<rect width="400" height="240" fill="url(#${sky})"/>`;

    switch (item.scene) {
      case 'beach':
        return `<svg viewBox="0 0 400 240" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${defs}${base}${sun}
          <path d="M0 140c60-15 115-15 180 0s150 12 220 0v100H0z" fill="url(#${sea})"/>
          <path d="M0 178c70-12 135-10 200 2s135 10 200-2v62H0z" fill="${c2}" opacity=".55"/>
          <path d="M0 201c60-11 112-9 172 2s142 9 228-2v39H0z" fill="#f7e6c2" opacity=".72"/>
          <path d="M52 204c22-5 38-5 55 0" stroke="#fff" stroke-width="3" fill="none" opacity=".55" stroke-linecap="round"/>
          <path d="M266 189c25-4 43-3 62 3" stroke="#fff" stroke-width="2.5" fill="none" opacity=".44" stroke-linecap="round"/>
          <g opacity=".62" fill="#0b4a43"><path d="M58 168c1-35 2-53 7-72" stroke="#0b4a43" stroke-width="3" fill="none"/><path d="M64 105c-18-14-28-4-31 5 12-5 23 1 31 8zm0 0c18-14 28-4 31 5-12-5-23 1-31 8z" /><path d="M64 105c-9-15-2-24 11-27-4 10-2 19-11 27z"/></g>
        </svg>`;
      case 'island':
        return `<svg viewBox="0 0 400 240" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${defs}${base}${sun}
          <rect y="142" width="400" height="98" fill="url(#${sea})"/>
          <ellipse cx="145" cy="153" rx="78" ry="20" fill="${c2}" opacity=".9"/>
          <ellipse cx="278" cy="164" rx="55" ry="14" fill="${c2}" opacity=".65"/>
          <path d="M145 148c-4-23-2-36 2-52" stroke="#123f3b" stroke-width="3.5" fill="none"/>
          <g fill="#123f3b" opacity=".8"><path d="M147 96c-18-10-27-1-29 8 12-5 21 0 29 7zm0 0c17-11 26-2 29 7-11-4-22 0-29 8z"/><path d="M147 96c-10-13-3-23 9-26-4 9-2 18-9 26z"/></g>
          <path d="M0 190c62-12 120-10 185 2s145 8 215-3" fill="none" stroke="#fff" stroke-width="3" opacity=".45"/>
          <path d="M0 211c72-8 130-6 195 4s135 7 205-2" fill="none" stroke="#fff" stroke-width="2" opacity=".3"/>
        </svg>`;
      case 'mountain':
        return `<svg viewBox="0 0 400 240" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${defs}${base}${sun}
          <path d="M-20 225 118 76l84 92 65-77 153 134v40H-20z" fill="${c2}" opacity=".78"/>
          <path d="M65 225 200 65l157 160v15H65z" fill="#163b42" opacity=".8"/>
          <path d="m118 76 49 50-20-5-14 18-17-21-18 2z" fill="#f7fbf8" opacity=".8"/>
          <path d="m200 65 55 56-22-6-15 20-18-24-18 5z" fill="#f7fbf8" opacity=".86"/>
          <path d="M0 205c64-12 121-9 192 4s145 8 208-3v34H0z" fill="${c2}" opacity=".92"/>
          <path d="M0 220c74-8 130-5 201 5s130 6 199-2" fill="none" stroke="#fff" stroke-width="2" opacity=".28"/>
        </svg>`;
      case 'city':
        return `<svg viewBox="0 0 400 240" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${defs}${base}${sun}
          <rect y="150" width="400" height="90" fill="${c2}" opacity=".55"/>
          <g fill="#123f3b" opacity=".8">
            <rect x="18" y="121" width="37" height="80"/><rect x="63" y="94" width="26" height="107"/>
            <rect x="96" y="132" width="44" height="69"/><rect x="148" y="82" width="32" height="119"/>
            <rect x="188" y="113" width="47" height="88"/><rect x="244" y="69" width="27" height="132"/>
            <rect x="278" y="121" width="43" height="80"/><rect x="329" y="101" width="29" height="100"/>
            <rect x="365" y="139" width="24" height="62"/>
          </g>
          <g fill="#f5e4b3" opacity=".7">
            <rect x="25" y="130" width="5" height="7"/><rect x="37" y="130" width="5" height="7"/><rect x="25" y="144" width="5" height="7"/>
            <rect x="155" y="94" width="5" height="7"/><rect x="165" y="94" width="5" height="7"/><rect x="155" y="108" width="5" height="7"/>
            <rect x="251" y="82" width="5" height="7"/><rect x="260" y="82" width="5" height="7"/><rect x="251" y="98" width="5" height="7"/>
            <rect x="288" y="134" width="5" height="7"/><rect x="300" y="134" width="5" height="7"/>
          </g>
          <path d="M0 202c80-8 142-5 212 3s128 6 188-2v37H0z" fill="#0a3330" opacity=".85"/>
          <path d="M40 223c35-4 65-4 95 0" stroke="#f5e4b3" stroke-width="2" opacity=".4" fill="none"/>
        </svg>`;
      case 'temple':
        return `<svg viewBox="0 0 400 240" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${defs}${base}${sun}
          <path d="M0 175c65-16 120-12 190 2s140 8 210-5v68H0z" fill="${c2}" opacity=".55"/>
          <g fill="#7b292e" opacity=".85">
            <rect x="132" y="88" width="7" height="100"/><rect x="260" y="88" width="7" height="100"/>
            <rect x="112" y="78" width="175" height="9" rx="3"/><rect x="121" y="98" width="157" height="7" rx="2"/>
            <rect x="196" y="78" width="7" height="31"/>
          </g>
          <g fill="#123f3b" opacity=".7">
            <circle cx="88" cy="134" r="26"/><circle cx="76" cy="150" r="21"/><rect x="85" y="150" width="5" height="38"/>
            <circle cx="316" cy="144" r="21"/><circle cx="328" cy="158" r="17"/><rect x="319" y="158" width="5" height="30"/>
          </g>
          <path d="M0 211c74-8 130-5 201 5s130 6 199-2v26H0z" fill="#f4e4c0" opacity=".35"/>
        </svg>`;
      case 'coast':
        return `<svg viewBox="0 0 400 240" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${defs}${base}${sun}
          <path d="M0 188 74 88l70 72 55-91 90 119v52H0z" fill="${c2}" opacity=".8"/>
          <path d="m181 69 38 55-16-4-12 17-14-20-16 4z" fill="#f7fbf8" opacity=".75"/>
          <rect y="172" width="400" height="68" fill="${c2}" opacity=".72"/>
          <path d="M0 198c58-13 110-11 172 1s142 10 228-2" fill="none" stroke="#fff" stroke-width="3" opacity=".45"/>
          <path d="M0 216c70-8 128-6 192 3s140 7 208-2" fill="none" stroke="#fff" stroke-width="2" opacity=".28"/>
        </svg>`;
      case 'park':
      default:
        return `<svg viewBox="0 0 400 240" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${defs}${base}${sun}
          <path d="M0 172c74-14 136-10 208 2s136 8 192-3v69H0z" fill="${c2}" opacity=".65"/>
          <circle cx="290" cy="125" r="52" fill="none" stroke="#f7fbf8" stroke-width="3" opacity=".7"/>
          <g stroke="#f7fbf8" stroke-width="2" opacity=".55">
            <path d="M290 73v104M238 125h104M253 88l74 74M327 88l-74 74"/>
          </g>
          <g fill="#f7fbf8" opacity=".82"><circle cx="290" cy="73" r="5"/><circle cx="342" cy="125" r="5"/><circle cx="290" cy="177" r="5"/><circle cx="238" cy="125" r="5"/></g>
          <path d="M290 177v22" stroke="#f7fbf8" stroke-width="3" opacity=".6"/>
          <path d="M0 204c70-8 130-5 200 5s130 6 200-2" fill="none" stroke="#fff" stroke-width="2" opacity=".3"/>
        </svg>`;
    }
  }

  function cardMarkup(item, index) {
    const highlighted = state.highlights.includes(item.id);
    const reasons = (item.match_reasons || []).slice(0, 3);
    const favorite = state.favorites.has(item.id);
    return `<article class="trip-card${highlighted ? ' is-highlight' : ''}" data-id="${esc(item.id)}" tabindex="0" role="button" aria-label="${esc(item.name)}" style="--i:${index}">
      <div class="card-visual" style="--c1:${esc(item.colors?.[0] || '')};--c2:${esc(item.colors?.[1] || '')}">
        ${sceneSvg(item)}
        <div class="card-topline">
          <span>${esc(item.country)}</span>
          <span class="card-rating">★ ${esc(item.rating)}</span>
        </div>
        ${highlighted ? `<span class="top-pick">${esc(t('card.topPick'))}</span>` : ''}
        <button class="card-heart" type="button" aria-label="Save ${esc(item.name)}" aria-pressed="${favorite}">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20.4 4.3 13a4.8 4.8 0 0 1 0-6.8 4.8 4.8 0 0 1 6.8 0l.9 1 .9-1a4.8 4.8 0 0 1 6.8 6.8z" fill="${favorite ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>
        </button>
      </div>
      <div class="card-body">
        <div class="card-heading"><div><h3 class="card-name">${esc(item.name)}</h3><p class="card-country">${esc(item.country)}</p></div></div>
        <p class="card-tagline">${esc(item.tagline)}</p>
        <div class="card-meta">
          <span class="meta-chip">${esc(nightsLabel(item.nights))}</span>
          <span class="meta-chip">${esc(climateLabel(item.climate))}</span>
          <span class="meta-chip">${esc(regionLabel(item.region))}</span>
        </div>
        <div class="match-row">${reasons.map((reason) => `<span class="match-chip">${esc(reason)}</span>`).join('')}</div>
        <div class="card-foot">
          <div class="price"><span class="price-label">${esc(t('card.from'))}</span><span class="price-value">${esc(money(item.price))}</span><span class="price-unit">${esc(t('card.perPerson'))}</span></div>
          <button class="view-btn" type="button">${esc(t('card.view'))}</button>
        </div>
      </div>
    </article>`;
  }

  function renderCards() {
    const cards = $('#cards');
    const empty = $('#emptyState');
    if (!cards) return;
    const items = state.items || [];
    cards.setAttribute('aria-busy', 'false');
    if (!items.length) {
      cards.innerHTML = '';
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    cards.innerHTML = items.map(cardMarkup).join('');
    cards.querySelectorAll('.trip-card').forEach((card) => {
      const itemId = card.dataset.id;
      card.addEventListener('click', (event) => {
        if (event.target.closest('.card-heart')) return;
        openModal(itemId);
      });
      card.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          openModal(itemId);
        }
      });
      card.querySelector('.card-heart')?.addEventListener('click', (event) => {
        event.stopPropagation();
        toggleFavorite(itemId);
      });
      card.querySelector('.view-btn')?.addEventListener('click', (event) => {
        event.stopPropagation();
        openModal(itemId);
      });
    });
  }

  function updateCatalog(payload) {
    if (payload.items) state.items = payload.items;
    if (payload.filters) state.filters = payload.filters;
    state.highlights = payload.highlights || [];
    state.total = payload.total == null ? state.items.length : payload.total;
    const count = $('#resultCount');
    if (count) count.textContent = String(state.total);
    renderFilters();
    renderCards();
    if (payload.relaxed && state.items.length) {
      // Keep the UI honest: the catalog was relaxed to the closest matches.
      const bar = $('#filterBar');
      if (bar && !bar.querySelector('.filter-relaxed')) {
        const note = document.createElement('span');
        note.className = 'filter-chip filter-relaxed is-soft';
        note.textContent = state.language === 'pt-BR' ? 'opções mais próximas' : 'closest matches';
        bar.appendChild(note);
      }
    }
  }

  function toggleFavorite(itemId) {
    if (state.favorites.has(itemId)) state.favorites.delete(itemId);
    else state.favorites.add(itemId);
    saveFavorites();
    renderCards();
    if (state.favorites.has(itemId)) toast(t('modal.saved'));
  }

  // ----------------------------------------------------------------- modal
  function openModal(itemId, { focus = true } = {}) {
    const item = state.items.find((entry) => entry.id === itemId);
    if (!item) return;
    state.modalItem = item;
    const modal = $('#tripModal');
    const content = $('#modalContent');
    if (!modal || !content) return;
    const highlights = (item.highlights || []).map((value) => `<li>${esc(value)}</li>`).join('');
    const bestFor = (item.best_for || []).map((value) => `<li>${esc(value)}</li>`).join('');
    content.innerHTML = `
      <div class="modal-hero" style="--c1:${esc(item.colors?.[0] || '')};--c2:${esc(item.colors?.[1] || '')}">
        ${sceneSvg(item)}
        <div class="modal-hero-overlay"></div>
        <div class="modal-hero-text"><p>${esc(item.country)}</p><h2 id="modalTitle">${esc(item.name)}</h2></div>
      </div>
      <div class="modal-body">
        <div class="modal-stats">
          <span class="modal-stat">★ ${esc(item.rating)} · ${esc(item.reviews)} ${esc(t('card.rating'))}</span>
          <span class="modal-stat">${esc(nightsLabel(item.nights))}</span>
          <span class="modal-stat">${esc(climateLabel(item.climate))}</span>
          <span class="modal-stat">${esc(regionLabel(item.region))}</span>
        </div>
        <p class="modal-summary">${esc(item.summary)}</p>
        <div class="modal-section"><h4>${esc(t('modal.highlights'))}</h4><ul class="modal-list">${highlights}</ul></div>
        <div class="modal-section"><h4>${esc(t('modal.bestFor'))}</h4><ul class="modal-list">${bestFor}</ul></div>
        <div class="modal-actions">
          <div class="modal-price">${esc(money(item.price))} <span>${esc(t('modal.priceNote'))}</span></div>
          <button class="primary-btn" id="askAbout" type="button">${esc(t('modal.ask'))}</button>
        </div>
      </div>`;
    modal.hidden = false;
    document.body.classList.add('modal-open');
    $('#askAbout')?.addEventListener('click', () => {
      closeModal();
      const prompt = state.language === 'pt-BR' ? `Me conte mais sobre ${item.name}` : `Tell me more about ${item.name}`;
      sendText(prompt);
    });
    if (focus) $('.modal-close')?.focus();
  }

  function closeModal() {
    const modal = $('#tripModal');
    if (!modal) return;
    modal.hidden = true;
    document.body.classList.remove('modal-open');
    state.modalItem = null;
  }

  // --------------------------------------------------------------- audio
  async function ensureAudio() {
    if (!state.audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      state.audioCtx = new Ctx({ latencyHint: 'interactive' });
    }
    if (state.audioCtx.state === 'suspended') {
      try { await state.audioCtx.resume(); } catch (error) { /* gesture required */ }
    }
    return state.audioCtx;
  }

  function requestWelcome(force = false) {
    if (!state.ready) return;
    if (state.status === 'thinking') return;
    if (state.status === 'speaking' && state.audioKind !== 'welcome') return;
    if (state.welcomeRequested && !force) return;
    if (state.welcomePlayed && !force) return;
    if (force && state.audioKind === 'welcome') return;
    if (state.micOn) state.suppressMic = true;
    state.welcomeRequested = true;
    clearTimeout(state.welcomeTimer);
    state.welcomeTimer = setTimeout(() => {
      if (!state.welcomePlayed && state.audioKind !== 'welcome') {
        state.welcomeRequested = false;
        state.suppressMic = false;
      }
    }, 6000);
    ensureAudio();
    send({ type: 'welcome' });
  }

  function finalizeWelcome() {
    clearTimeout(state.welcomeTimer);
    state.welcomeTimer = null;
    if (state.audioKind === 'welcome' || state.welcomeRequested) {
      state.welcomePlayed = true;
      state.welcomeRequested = false;
    }
    state.audioKind = '';
    state.suppressMic = false;
  }

  async function beginVoice() {
    if (!state.ready) {
      toast(t('toast.notReady'));
      return;
    }
    const needWelcome = !state.welcomePlayed && !state.welcomeRequested;
    const welcomePending = !state.welcomePlayed;
    if (welcomePending) state.suppressMic = true;
    if (!state.micOn) await startMic();
    if (needWelcome) requestWelcome();
  }

  async function playPcm(arrayBuffer) {
    const ctx = await ensureAudio();
    if (!ctx) return;
    const samples = new Float32Array(arrayBuffer);
    if (!samples.length) return;

    let sum = 0;
    const stride = Math.max(1, Math.floor(samples.length / 80));
    for (let i = 0; i < samples.length; i += stride) sum += samples[i] * samples[i];
    updateWaveLevel(Math.sqrt(sum / Math.ceil(samples.length / stride)) * 1.2);

    const buffer = ctx.createBuffer(1, samples.length, 24000);
    buffer.copyToChannel(samples, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    let startAt = state.nextPlayTime;
    if (!startAt || startAt < now + 0.02) startAt = now + 0.03;
    // Phonon generates faster than real time, so it is normal for the queue to
    // run several seconds ahead.  Do not clamp it: clamping would overlap
    // sources and clip the assistant's reply.
    try { source.start(startAt); } catch (error) { source.start(); }
    state.nextPlayTime = startAt + buffer.duration;
    state.sources.push(source);
    source.onended = () => {
      state.sources = state.sources.filter((entry) => entry !== source);
      maybePlaybackDone();
    };
    // Safety: onended is reliable, but a suspended context can swallow it.
    clearTimeout(state.audioEndTimer);
    state.audioEndTimer = setTimeout(maybePlaybackDone, Math.max(400, (state.nextPlayTime - ctx.currentTime + 1.4) * 1000));
  }

  function maybePlaybackDone() {
    if (state.pendingAudioEnd == null) return;
    if (state.sources.length > 0) return;
    const seq = state.pendingAudioEnd;
    state.pendingAudioEnd = null;
    clearTimeout(state.audioEndTimer);
    state.audioEndTimer = null;
    state.nextPlayTime = 0;
    send({ type: 'playback_done', seq });
    const kind = state.audioKind;
    if (kind === 'welcome') finalizeWelcome();
    else state.audioKind = '';
  }

  function stopAudio() {
    for (const source of state.sources) {
      try { source.stop(); } catch (error) { /* already stopped */ }
    }
    state.sources = [];
    state.nextPlayTime = 0;
    state.pendingAudioEnd = null;
    state.audioSeq = null;
    clearTimeout(state.audioEndTimer);
    state.audioEndTimer = null;
  }

  function updateWaveLevel(level) {
    const now = performance.now();
    if (now - state.lastWaveUpdate < 45) return;
    state.lastWaveUpdate = now;
    state.level = clamp(0.72 * state.level + 0.28 * level, 0, 1);
    const bars = state.waveBars;
    if (!bars.length) return;
    for (let i = 0; i < bars.length; i += 1) {
      const shape = 0.35 + 0.65 * Math.abs(Math.sin(i * 0.78 + now / 260));
      const height = 3 + state.level * 19 * shape;
      bars[i].style.height = `${height.toFixed(1)}px`;
    }
  }

  function resetWaveform() {
    state.level = 0;
    state.waveBars.forEach((bar) => { bar.style.height = '3px'; });
  }

  // ------------------------------------------------------------------ mic
  async function startMic() {
    if (!state.ready) {
      toast(t('toast.notReady'));
      return;
    }
    if (state.micOn) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      toast(t('toast.micUnsupported'), true);
      return;
    }
    const ctx = await ensureAudio();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 24000,
        },
      });
      const capture = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 24000,
        latencyHint: 'interactive',
      });
      if (capture.state === 'suspended') await capture.resume();
      await capture.audioWorklet.addModule('/audio-worklet.js?v=10');
      const source = capture.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(capture, 'onda-pcm24k');
      state.micFrameCount = 0;
      node.port.onmessage = (event) => {
        const buffer = event.data;
        if (!(buffer instanceof ArrayBuffer)) return;
        state.micFrameCount += 1;
        if (state.micFrameCount === 1) {
          clearTimeout(state.micFrameTimer);
          state.micFrameTimer = null;
        }
        // The welcome greeting is played through the speakers while the mic is
        // already open (keeping the browser's user gesture).  Do not send its
        // echo to ASR; resume frames when the welcome audio drains.
        if (state.suppressMic) return;
        const samples = new Float32Array(buffer);
        let sum = 0;
        for (let i = 0; i < samples.length; i += 4) sum += samples[i] * samples[i];
        updateWaveLevel(Math.sqrt(sum / Math.max(1, samples.length / 4)) * 14);
        if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(buffer);
      };
      const sink = capture.createGain();
      sink.gain.value = 0;
      source.connect(node);
      node.connect(sink);
      sink.connect(capture.destination);

      state.micCtx = capture;
      state.micStream = stream;
      state.micNode = node;
      state.micSource = source;
      state.micOn = true;
      send({ type: 'start' });
      setStatus('listening');
      toast(t('toast.micOn'));
      clearTimeout(state.micFrameTimer);
      state.micFrameTimer = setTimeout(() => {
        if (state.micOn && state.micFrameCount === 0) {
          toast(t('toast.noMicAudio'), true);
        }
      }, 2600);
      if (ctx && ctx.state === 'suspended') await ctx.resume();
    } catch (error) {
      console.warn('mic error', error);
      const denied = error && (error.name === 'NotAllowedError' || error.name === 'SecurityError');
      toast(denied ? t('toast.micDenied') : t('toast.micUnsupported'), true);
      state.micOn = false;
      state.suppressMic = false;
      setStatus('idle');
    }
  }

  async function stopMic({ silent = false, send: sendStop = true } = {}) {
    if (sendStop) send({ type: 'stop' });
    state.micOn = false;
    state.suppressMic = false;
    clearTimeout(state.micFrameTimer);
    state.micFrameTimer = null;
    try { state.micNode?.disconnect(); } catch (error) { /* noop */ }
    try { state.micSource?.disconnect(); } catch (error) { /* noop */ }
    try { state.micStream?.getTracks().forEach((track) => track.stop()); } catch (error) { /* noop */ }
    try { await state.micCtx?.close(); } catch (error) { /* noop */ }
    state.micNode = null;
    state.micSource = null;
    state.micStream = null;
    state.micCtx = null;
    resetWaveform();
    if (!silent) {
      toast(t('toast.micOff'));
      setStatus('idle');
    }
  }

  function toggleMic() {
    if (state.audioKind === 'welcome' || (state.welcomeRequested && !state.welcomePlayed)) {
      // Clicking during the welcome means "skip it and listen".
      send({ type: 'stop_audio' });
      stopAudio();
      finalizeWelcome();
      if (!state.micOn) beginVoice();
      else setStatus('listening');
      return;
    }
    if (state.status === 'speaking' || state.status === 'thinking') {
      send({ type: 'stop_audio' });
      stopAudio();
      state.audioKind = '';
      setStatus('listening');
      return;
    }
    if (state.micOn) stopMic();
    else beginVoice();
  }

  // --------------------------------------------------------------- sending
  function sendText(text) {
    const value = String(text || '').trim();
    if (!value) return;
    if (!state.ready || !state.connected) {
      toast(t('toast.notReady'));
      return;
    }
    if (state.audioKind === 'welcome' || (state.welcomeRequested && !state.welcomePlayed)) {
      send({ type: 'stop_audio' });
      stopAudio();
      finalizeWelcome();
    }
    // A typed message is an explicit choice to start the conversation, not to
    // listen to the greeting first.
    state.welcomePlayed = true;
    state.welcomeRequested = false;
    state.suppressMic = false;
    if (state.status === 'speaking' || state.status === 'thinking') {
      send({ type: 'stop_audio' });
      stopAudio();
      state.audioKind = '';
    }
    send({ type: 'text', text: value });
    setStatus('thinking');
    const input = $('#textInput');
    if (input) input.value = '';
    ensureAudio();
  }

  // ------------------------------------------------------------- messages
  function handleMessage(message) {
    switch (message.type) {
      case 'ready': {
        state.ready = true;
        state.capabilities = message.capabilities || {};
        state.serverWarnings = Array.isArray(message.warnings) ? message.warnings : [];
        updateCatalog({
          items: message.catalog || [],
          filters: message.filters || {},
          highlights: [],
          total: (message.catalog || []).length,
          relaxed: false,
        });
        hideBoot();
        renderTelemetry();
        const preferred = state.language;
        const serverLang = message.language === 'pt-BR' ? 'pt-BR' : 'en';
        applyLanguage(preferred, { notifyServer: false });
        if (preferred !== serverLang) send({ type: 'language', language: preferred });
        state.messages = [];
        state.pendingUser = '';
        state.pendingAssistant = null;
        state.welcomePlayed = false;
        state.welcomeRequested = false;
        clearTimeout(state.welcomeTimer);
        state.welcomeTimer = null;
        state.suppressMic = false;
        state.audioKind = '';
        addMessage('assistant', t('agent.greeting'), { greeting: true });
        renderMessages();
        setStatus('idle');
        if (!state.capabilities.stt) {
          const mic = $('#micBtn');
          const heroMic = $('#heroMic');
          if (mic) mic.disabled = true;
          if (heroMic) heroMic.disabled = true;
        }
        if (!state.capabilities.tts) {
          // Text-only is a valid demo mode; no banner needed, but keep the
          // telemetry honest.
          renderTelemetry();
        }
        setTimeout(() => toast(t('boot.ready')), 500);
        break;
      }
      case 'state': {
        if (message.state === 'listening' && !state.micOn) {
          setStatus('idle');
        } else {
          setStatus(message.state || 'idle');
        }
        break;
      }
      case 'mic': {
        state.micOn = Boolean(message.on);
        setStatus(state.micOn ? 'listening' : 'idle');
        break;
      }
      case 'user_partial': {
        state.pendingUser = message.text || '';
        renderMessages();
        break;
      }
      case 'user_final': {
        const text = String(message.text || '').trim();
        if (text) {
          commitAssistant();
          addMessage('user', text, { input: message.input || 'voice' });
          state.pendingUser = '';
          renderMessages();
        }
        break;
      }
      case 'user_backchannel': {
        // A short acknowledgement does not take the floor; nothing to render.
        break;
      }
      case 'assistant_thinking': {
        commitAssistant();
        state.pendingAssistant = { text: '', typing: true };
        setStatus('thinking');
        renderMessages();
        break;
      }
      case 'assistant_start': {
        if (!state.pendingAssistant) state.pendingAssistant = { text: '', typing: true };
        state.pendingAssistant.typing = false;
        setStatus('thinking');
        renderMessages();
        break;
      }
      case 'assistant_text': {
        if (!state.pendingAssistant) state.pendingAssistant = { text: '', typing: false };
        state.pendingAssistant.typing = false;
        state.pendingAssistant.text = message.text || '';
        renderMessages();
        break;
      }
      case 'assistant_interrupted': {
        if (state.pendingAssistant) state.pendingAssistant.interrupted = true;
        commitAssistant();
        renderMessages();
        break;
      }
      case 'assistant_end': {
        commitAssistant();
        renderMessages();
        break;
      }
      case 'language': {
        const language = message.language === 'pt-BR' ? 'pt-BR' : 'en';
        if (language !== state.language) {
          applyLanguage(language, { notifyServer: false });
        }
        break;
      }
      case 'catalog': {
        updateCatalog(message);
        break;
      }
      case 'policy': {
        state.lastPolicy = message;
        renderTelemetry();
        break;
      }
      case 'turn': {
        state.lastTurnForced = Boolean(message.forced);
        state.lastIntent = message.intent || '';
        renderTelemetry();
        break;
      }
      case 'welcome': {
        const greeting = state.messages.find((entry) => entry.meta?.greeting);
        if (greeting) greeting.text = message.text || t('agent.greeting');
        else addMessage('assistant', message.text || t('agent.greeting'), { greeting: true });
        renderMessages();
        break;
      }
      case 'audio_start': {
        state.audioSeq = message.seq;
        state.audioKind = message.kind || 'answer';
        if (state.audioKind === 'welcome') {
          state.welcomeRequested = true;
          clearTimeout(state.welcomeTimer);
          state.welcomeTimer = null;
          if (state.micOn) state.suppressMic = true;
        }
        state.pendingAudioEnd = null;
        state.nextPlayTime = 0;
        setStatus('speaking');
        ensureAudio();
        break;
      }
      case 'audio_end': {
        if (message.seq != null && state.audioSeq != null && message.seq !== state.audioSeq) break;
        state.pendingAudioEnd = message.seq;
        maybePlaybackDone();
        break;
      }
      case 'audio_control': {
        if (message.action === 'stop') {
          const wasWelcome = state.audioKind === 'welcome';
          stopAudio();
          if (wasWelcome) finalizeWelcome();
          else state.audioKind = '';
          if (state.status === 'speaking') setStatus(state.micOn ? 'listening' : 'idle');
        }
        break;
      }
      case 'reset': {
        state.messages = [];
        state.pendingUser = '';
        state.pendingAssistant = null;
        state.filters = message.filters || state.filters;
        state.highlights = [];
        state.lastPolicy = null;
        state.lastTurnForced = false;
        state.lastIntent = "";
        state.welcomePlayed = false;
        state.welcomeRequested = false;
        clearTimeout(state.welcomeTimer);
        state.welcomeTimer = null;
        state.suppressMic = false;
        state.audioKind = '';
        addMessage('assistant', t('agent.greeting'), { greeting: true });
        renderMessages();
        renderTelemetry();
        setStatus(state.micOn ? 'listening' : 'idle');
        break;
      }
      case 'error': {
        toast(message.message || t('toast.error'), true);
        if (message.fatal) {
          bootError(message.message || t('toast.error'));
          const mic = $('#micBtn');
          if (mic) mic.disabled = true;
        }
        break;
      }
      case 'pong': break;
      default: break;
    }
  }

  // ------------------------------------------------------------- telemetry
  function renderTelemetry() {
    const panel = $('#telemetry');
    if (!panel) return;
    const caps = state.capabilities || {};
    const mode = $('#telemetryMode');
    if (mode) {
      const stream = caps.laya_stream ? `${caps.laya_stream} stream` : 'no policy';
      const model = caps.policy ? (caps.policy_model || 'Laya') : t('telemetry.noPolicy');
      mode.textContent = `${stream} · ${model}`;
    }
    const policy = state.lastPolicy;
    const tag = $('#lastTag');
    const latency = $('#lastLatency');
    if (tag) tag.textContent = policy ? policy.tag : '—';
    if (latency) latency.textContent = policy ? `${policy.latency_ms} ms` : '—';
    const intent = $('#lastIntent');
    if (intent) intent.textContent = state.lastIntent || '—';
    const bars = $('#probBars');
    if (bars) {
      bars.innerHTML = '';
      if (policy && policy.probs) {
        const entries = Object.entries(policy.probs).sort((a, b) => b[1] - a[1]).slice(0, 4);
        for (const [name, value] of entries) {
          const row = document.createElement('div');
          row.className = 'prob-row';
          row.innerHTML = `<span class="prob-label">${esc(name.replace('user_', ''))}</span><span class="prob-track"><span class="prob-fill" style="width:${Math.round(value * 100)}%"></span></span><span>${Math.round(value * 100)}</span>`;
          bars.appendChild(row);
        }
      }
    }
    const models = $('#telemetryModels');
    if (models) {
      const parts = [];
      if (caps.policy_model) parts.push(`policy ${caps.policy_model}`);
      if (caps.llm_model) parts.push(`llm ${caps.llm_model}`);
      if (caps.tts_model) parts.push(`tts ${caps.tts_model}`);
      if (!parts.length) parts.push('text-only mode');
      if (state.lastTurnForced) parts.push('watchdog answer');
      if (state.serverWarnings.length) parts.push(`warnings: ${state.serverWarnings.join(' | ')}`);
      models.textContent = parts.join(' · ');
      models.title = state.serverWarnings.join('\n');
    }
  }

  function toggleTelemetry() {
    const panel = $('#telemetry');
    if (!panel) return;
    panel.hidden = !panel.hidden;
    const pressed = panel.hidden ? 'false' : 'true';
    ['#debugToggle', '#debugToggleAlt'].forEach((selector) => {
      const button = $(selector);
      if (button) button.setAttribute('aria-pressed', pressed);
    });
    renderTelemetry();
  }

  // ---------------------------------------------------------------- reset
  function resetSession() {
    if (!state.ready) return;
    stopAudio();
    send({ type: 'reset' });
    state.pendingUser = '';
    state.pendingAssistant = null;
    state.messages = [];
    state.lastPolicy = null;
    state.lastTurnForced = false;
    state.lastIntent = "";
    addMessage('assistant', t('agent.greeting'), { greeting: true });
    renderMessages();
    setStatus(state.micOn ? 'listening' : 'idle');
  }

  // ----------------------------------------------------------------- init
  function init() {
    state.waveBars = $$('#waveform span');
    applyLanguage(state.language);
    renderTelemetry();

    $$('.lang-btn').forEach((button) => {
      button.addEventListener('click', () => {
        const language = button.dataset.lang === 'pt-BR' ? 'pt-BR' : 'en';
        if (language === state.language) return;
        applyLanguage(language, { notifyServer: true, showToast: true });
      });
    });

    $('#heroMic')?.addEventListener('click', async () => {
      await ensureAudio();
      $('#agentPanel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setTimeout(() => {
        if (!state.micOn) beginVoice();
      }, 320);
    });
    $('#micBtn')?.addEventListener('click', toggleMic);
    $('#pauseBtn')?.addEventListener('click', () => {
      if (state.micOn) stopMic();
      else beginVoice();
    });
    $('#resetBtn')?.addEventListener('click', resetSession);
    $('#debugToggle')?.addEventListener('click', toggleTelemetry);
    $('#debugToggleAlt')?.addEventListener('click', toggleTelemetry);
    $('#clearFilters')?.addEventListener('click', clearFilters);
    $('#emptyClear')?.addEventListener('click', clearFilters);
    $('#emptyAsk')?.addEventListener('click', () => {
      const prompt = state.language === 'pt-BR'
        ? 'Me sugira algumas viagens que combinem com o que eu gosto'
        : 'Suggest a few trips that match what I like';
      sendText(prompt);
    });

    $('#composer')?.addEventListener('submit', (event) => {
      event.preventDefault();
      sendText($('#textInput')?.value || '');
    });

    const firstGestureWelcome = (event) => {
      if (!state.ready || state.welcomePlayed || state.welcomeRequested) return;
      if (event.target.closest?.('#composer, .suggestion, #micBtn, #heroMic, .lang-switch, input, textarea')) return;
      ensureAudio();
      requestWelcome();
    };
    document.addEventListener('pointerdown', firstGestureWelcome, { once: true });
    document.addEventListener('keydown', firstGestureWelcome, { once: true });

    $$('[data-close-modal]').forEach((element) => element.addEventListener('click', closeModal));
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closeModal();
    });

    const input = $('#textInput');
    if (input) {
      input.addEventListener('focus', () => { ensureAudio(); });
    }

    window.addEventListener('beforeunload', () => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.close();
      }
    });

    connect();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
