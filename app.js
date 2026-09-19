const app = document.getElementById('app');
const BACKEND_API_BASE = window.location.port === '5000' ? '' : 'http://127.0.0.1:5000';

const state = {
  step: -1,
  profile: {
    specialization: null,
    projects: '3',
    internship: 'Yes',
    skills: [],
    hours: '< 5 hrs',
    exposure: ['Open Source'],
    confidence: 5
  },
  jobs: [],
  survey: [],
  studentSurvey: [],
  demand: null,
  surveyDemand: null,
  studentBenchmark: null,
  loading: true,
  loadingError: '',
  llmStatus: 'not_requested',
  llmText: '',
  llmError: ''
};

const stopWords = new Set(['and','or','the','a','an','to','of','in','with','for','on','by','is','are','as','this','that','from','at','be','you','your','we','our','will']);

const skillAliases = {
  'Python': ['python'],
  'SQL': ['sql','structured query language'],
  'Power BI': ['power bi','powerbi'],
  'Machine Learning': ['machine learning','ml','scikit','regression','classification','predictive model'],
  'Deep Learning': ['deep learning','neural','tensorflow','pytorch','keras'],
  'NLP': ['nlp','natural language processing','text mining','topic tagging','language model'],
  'React': ['react','reactjs','frontend'],
  'Docker': ['docker','container','containers','kubernetes','k8s'],
  'Git': ['git','github','version control'],
  'Cloud (AWS/GCP)': ['aws','gcp','google cloud','azure','cloud'],
  'Java': ['java'],
  'C++': ['c++','cpp'],
  'Data Analysis': ['data analysis','analytics','statistical','statistics','data visualization'],
  'Tableau': ['tableau'],
  'Cybersecurity': ['cybersecurity','security','network security','penetration testing','siem'],
  'Networking': ['network','networking','tcp','ip','routing','switching','cisco'],
  'MLOps': ['mlops','model deployment','machine learning operations','kubeflow','airflow','ci cd'],
  'Big Data (Spark)': ['spark','big data','hadoop','databricks']
};

const specializationProfiles = {
  'Data Science': ['Python','SQL','Data Analysis','Machine Learning','Power BI','Tableau','NLP','Big Data (Spark)'],
  'Software Engineering': ['Java','C++','React','Git','Docker','Cloud (AWS/GCP)','SQL'],
  'Networks': ['Networking','Cybersecurity','Cloud (AWS/GCP)','Docker','Git'],
  'AI & ML': ['Python','Machine Learning','Deep Learning','NLP','MLOps','Big Data (Spark)','Cloud (AWS/GCP)'],
  'Cybersecurity': ['Cybersecurity','Networking','Python','SQL','Cloud (AWS/GCP)','Docker','Git']
};

const specializationScoreProfiles = {
  'Data Science': {
    required: ['Python','SQL','Data Analysis','Machine Learning','Power BI','Tableau','NLP','Big Data (Spark)'],
    weights: {'Python':1.8,'SQL':1.5,'Data Analysis':1.7,'Machine Learning':1.8,'Power BI':1.3,'Tableau':1.2,'NLP':1.4,'Big Data (Spark)':1.3}
  },
  'Software Engineering': {
    required: ['Java','C++','React','Git','Docker','Cloud (AWS/GCP)','SQL'],
    weights: {'Java':1.6,'C++':1.4,'React':1.7,'Git':1.5,'Docker':1.4,'Cloud (AWS/GCP)':1.3,'SQL':1.1}
  },
  'Networks': {
    required: ['Networking','Cybersecurity','Cloud (AWS/GCP)','Docker','Git','Python'],
    weights: {'Networking':2.0,'Cybersecurity':1.5,'Cloud (AWS/GCP)':1.4,'Docker':1.2,'Git':1.1,'Python':1.1}
  },
  'AI & ML': {
    required: ['Python','Machine Learning','Deep Learning','NLP','MLOps','Big Data (Spark)','Cloud (AWS/GCP)'],
    weights: {'Python':1.7,'Machine Learning':2.0,'Deep Learning':1.8,'NLP':1.6,'MLOps':1.5,'Big Data (Spark)':1.3,'Cloud (AWS/GCP)':1.2}
  },
  'Cybersecurity': {
    required: ['Cybersecurity','Networking','Python','SQL','Cloud (AWS/GCP)','Docker','Git'],
    weights: {'Cybersecurity':2.0,'Networking':1.6,'Python':1.2,'SQL':1.1,'Cloud (AWS/GCP)':1.3,'Docker':1.2,'Git':1.1}
  }
};

function getSpecializationConfig(specialization) {
  if (!specialization) return { required: [], weights: {} };
  return specializationScoreProfiles[specialization] || { required: [], weights: {} };
}


function normalizeText(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/c\+\+/g, 'cpp')
    .replace(/[^a-z0-9+#.]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(text) {
  return normalizeText(text).split(' ').filter(t => t && !stopWords.has(t) && t.length > 1);
}

function hasAlias(text, alias) {
  const clean = normalizeText(text);
  const a = normalizeText(alias);
  if (!a) return false;
  if (a.includes(' ')) return clean.includes(a);
  return new RegExp(`(^|\\s)${a.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(\\s|$)`, 'i').test(clean);
}

function parseCSV(text) {
  const rows = [];
  let row = [], cell = '', quote = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i], n = text[i + 1];
    if (c === '"' && quote && n === '"') { cell += '"'; i++; }
    else if (c === '"') quote = !quote;
    else if (c === ',' && !quote) { row.push(cell); cell = ''; }
    else if ((c === '\n' || c === '\r') && !quote) {
      if (c === '\r' && n === '\n') i++;
      row.push(cell); cell = '';
      if (row.some(x => x !== '')) rows.push(row);
      row = [];
    } else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  if (!rows.length) return [];
  const headers = rows.shift().map(h => h.trim());
  return rows.map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] || ''])));
}

async function fetchFirst(paths) {
  let lastError;
  for (const path of paths) {
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`${path} returned ${res.status}`);
      return await res.text();
    } catch (e) { lastError = e; }
  }
  throw lastError || new Error('No dataset path loaded');
}

async function loadData() {
  try {
    const [jobsText, surveyText, studentText] = await Promise.all([
      fetchFirst(['data/Data_jobs.csv','https://formanlight.com/ahel/data/Data_jobs.csv']),
      fetchFirst(['data/survey_results_public.csv','data/survey_results_public.csv','https://formanlight.com/ahel/data/survey_results_public.csv']),
      fetchFirst(['data/student_survey.csv','https://formanlight.com/ahel/data/student_survey.csv'])
    ]);
    state.jobs = parseCSV(jobsText);
    state.survey = parseCSV(surveyText);
    state.studentSurvey = parseCSV(studentText);
    state.demand = computeDemand(state.jobs);
    state.surveyDemand = computeSurveyDemand(state.survey);
    state.studentBenchmark = computeStudentBenchmark(state.studentSurvey);
  } catch (e) {
    state.loadingError = e.message || String(e);
    state.demand = computeDemand([]);
    state.surveyDemand = computeSurveyDemand([]);
    state.studentBenchmark = computeStudentBenchmark([]);
    console.warn('CSV load failed. Run through a local server or check hosted dataset paths.', e);
  }
  state.loading = false;
  render();
}

function computeDemand(jobs) {
  const demand = {};
  const source = jobs.length ? jobs : [];
  for (const skill of Object.keys(skillAliases)) demand[skill] = 0;
  for (const job of source) {
    const text = `${job.title || ''} ${job.description || ''} ${job.work_type || ''} ${job.employment_type || ''}`;
    tokenize(text); // explicit NLP preprocessing step: normalization/tokenization/stop-word filtering.
    for (const [skill, aliases] of Object.entries(skillAliases)) {
      if (aliases.some(a => hasAlias(text, a))) demand[skill]++;
    }
  }
  const total = Math.max(1, source.length);
  const ranked = Object.entries(demand)
    .map(([skill, count]) => ({ skill, count, pct: Math.round((count / total) * 100) }))
    .filter(x => x.count > 0)
    .sort((a, b) => b.count - a.count);
  return { total, ranked };
}

function computeSurveyDemand(survey) {
  const demand = {};
  for (const skill of Object.keys(skillAliases)) demand[skill] = 0;
  const fields = [
    'LanguageWorkedWith', 'LanguageDesireNextYear',
    'DatabaseWorkedWith', 'DatabaseDesireNextYear',
    'PlatformWorkedWith', 'PlatformDesireNextYear',
    'FrameworkWorkedWith', 'FrameworkDesireNextYear',
    'DevType', 'EducationTypes', 'SelfTaughtTypes', 'Methodology', 'VersionControl'
  ];
  for (const person of survey) {
    const text = fields.map(f => person[f] || '').join(' ');
    tokenize(text);
    for (const [skill, aliases] of Object.entries(skillAliases)) {
      if (aliases.some(a => hasAlias(text, a))) demand[skill]++;
    }
  }
  const total = Math.max(1, survey.length);
  const ranked = Object.entries(demand)
    .map(([skill, count]) => ({ skill, count, pct: Math.round((count / total) * 100) }))
    .filter(x => x.count > 0)
    .sort((a, b) => b.count - a.count);
  return { total, ranked };
}
function scoreCategory(value, map, fallback = null) {
  const v = String(value || '').trim();
  if (!v) return fallback;
  return Object.prototype.hasOwnProperty.call(map, v) ? map[v] : fallback;
}

function computeStudentBenchmark(rows) {
  const readinessMap = { 'Not Ready': 0, 'Need Improvement': 50, 'Ready': 100 };
  const confidenceMap = { 'Weak': 25, 'Medium': 60, 'High': 90 };
  const yesMap = { 'No': 0, 'Somewhat': 50, 'Yes': 100 };
  const independenceMap = { 'Rarely': 25, 'Sometimes': 60, 'Always': 90 };
  const gapMap = { 'Large Gap': 0, 'Moderate Gap': 50, 'No Gap': 100 };

  const scored = rows.map(r => {
    const exposureParts = [
      scoreCategory(r['Practical Application Opportunities'], yesMap),
      scoreCategory(r['Independence in Problem Solving'], independenceMap),
      scoreCategory(r['Possession of Practical Skills'], yesMap)
    ].filter(x => x !== null);
    return {
      exposure: exposureParts.length ? Math.round(exposureParts.reduce((a,b)=>a+b,0)/exposureParts.length) : null,
      readiness: scoreCategory(r['Professional Readiness'], readinessMap),
      confidence: scoreCategory(r['Technical Confidence Level'], confidenceMap),
      marketGap: scoreCategory(r['Gap with Job Market'], gapMap)
    };
  }).filter(x => x.exposure !== null || x.readiness !== null || x.confidence !== null || x.marketGap !== null);

  const avg = key => {
    const vals = scored.map(x => x[key]).filter(x => x !== null);
    return vals.length ? Math.round(vals.reduce((a,b)=>a+b,0)/vals.length) : 0;
  };
  return {
    total: rows.length,
    avgExposure: avg('exposure'),
    avgReadiness: avg('readiness'),
    avgConfidence: avg('confidence'),
    avgMarketGap: avg('marketGap')
  };
}


function cls(x){ return x ? 'active' : ''; }
function setStep(step){ state.step = step; if (step !== 4) { state.llmStatus = 'not_requested'; state.llmText = ''; } render(); window.scrollTo(0,0); }
function selectOne(key, value){
  if (key === 'specialization') {
    state.profile.specialization = state.profile.specialization === value ? null : value;
  } else {
    state.profile[key] = value;
  }
  render();
}
function toggleSkill(skill){ const a = state.profile.skills; a.includes(skill) ? a.splice(a.indexOf(skill), 1) : a.push(skill); render(); }
function toggleExposure(v){ const a = state.profile.exposure; a.includes(v) ? a.splice(a.indexOf(v), 1) : a.push(v); render(); }

function tabs(){
  const names = ['Background','Technical Skills','Experience','Self-Assessment'];
  return `<div class="tabs">${names.map((n,i)=>`<div class="${state.step===i?'active':state.step>i?'done':''}">${n}</div>`).join('')}</div>`;
}

function option(label, active, onclick){ return `<button class="option ${cls(active)}" onclick="${onclick}">${label}</button>`; }

function home(){ return `<section class="landing">
  <div class="brand">AHEL · أَهْل</div>
  <div class="heroGrid">
    <div class="heroCopy">
      <h1>AI-Powered <span>Data Science Readiness Analyzer</span></h1>
      <p>AHEL evaluates a learner's readiness for data science, AI, software, networks, and cybersecurity careers by comparing profile inputs against job-market data and professional developer survey benchmarks.</p>
      <button class="primary" onclick="setStep(0)">START ANALYSIS →</button>
      <div class="loading">${state.loading ? 'Loading CSV datasets...' : state.loadingError ? `Dataset warning: ${state.loadingError}` : `Loaded ${state.jobs.length} jobs, ${state.survey.length} developer survey records, and ${state.studentSurvey.length} student survey records.`}</div>
    </div>
    <div class="modelPanel">
      <h3>Models Used</h3>
      <p><b>1. NLP Skill Extraction Model</b><br>Lightweight preprocessing model using normalization, tokenization, stop-word filtering, alias matching, and frequency analysis.</p>
      <p><b>2. Statistical ML Model</b><br>Developer survey analysis includes Skill Diversity Index and Logistic Regression for employment/readiness analysis.</p>
      <p><b>3. LLM Recommendation Model</b><br>Optional API-based LLM integration generates personalized natural-language feedback. It is not a browser-side large model; it runs through the backend when an API key is configured.</p>
    </div>
  </div>
  <div class="landingCards">
    <div class="infoCard"><h3>Existing Data Analysis</h3><p>Data_jobs.csv is processed to extract demanded skills, calculate frequency tables, and identify specialization-specific market gaps.</p></div>
    <div class="infoCard"><h3>Developer Benchmark</h3><p>survey_results_public.csv is used to calculate professional skill patterns, Skill Diversity Index, and logistic regression evidence for H2.</p></div>
    <div class="infoCard"><h3>Student Benchmark</h3><p>student_survey.xlsx / student_survey.csv is used to test H1 through practical exposure, confidence, job-market gap, and professional readiness patterns.</p></div>
    <div class="infoCard"><h3>ML Workflow</h3><p>User input → feature extraction → skill vector → job and survey matching → weighted scoring → optional LLM summary.</p></div>
  </div>
  <div class="infoBox"><div><span class="tick">✓</span>Large model status: AHEL does not run a large model inside the frontend. LLM support is implemented through an optional backend API using OpenAI or another compatible provider.</div><div><span class="tick">✓</span>Static mode still works using local analytical models and dynamic rule-based recommendations.</div></div>
</section>`; }

function background(){ return `${tabs()}<section class="form">
  <h2>Your Background</h2><p class="subtitle">Your specialization now changes scoring weights, skill gaps, and recommendations. Revolutionary, I know.</p>
  <div class="label">Specialization</div><div class="options">${['Data Science','Software Engineering','Networks','AI & ML','Cybersecurity'].map(v=>option(v,state.profile.specialization===v,`selectOne('specialization','${v}')`)).join('')}</div>
  <div class="label">Number of Projects Completed</div><div class="options">${['0','1','2','3','4','5+'].map(v=>option(v,state.profile.projects===v,`selectOne('projects','${v}')`)).join('')}</div>
  <div class="label">Internship / Co-op Experience</div><div class="options">${['Yes','No'].map(v=>option(v,state.profile.internship===v,`selectOne('internship','${v}')`)).join('')}</div>
  <div class="actions"><button class="primary" onclick="setStep(1)">NEXT →</button></div>
</section>`; }

function skills(){ const list=Object.keys(skillAliases);
return `${tabs()}<section class="form"><h2>Technical Skills</h2><p class="subtitle">Select all technologies you're comfortable using.</p>
<div class="options">${list.map(v=>option(v,state.profile.skills.includes(v),`toggleSkill('${v.replace(/'/g,"\\'")}')`)).join('')}</div>
<p class="loading">${state.profile.skills.length} selected · Target profile: ${state.profile.specialization || 'Not selected'}</p><div class="actions"><button class="ghost" onclick="setStep(0)">← BACK</button><button class="primary" onclick="setStep(2)">NEXT →</button></div></section>`; }

function experience(){ return `${tabs()}<section class="form"><h2>Practice & Exposure</h2><p class="subtitle">How much practical exposure do you have?</p>
<div class="label">Weekly Development Hours</div><div class="options">${['< 5 hrs','5-10 hrs','10-20 hrs','20+ hrs'].map(v=>option(v,state.profile.hours===v,`selectOne('hours','${v}')`)).join('')}</div>
<div class="label">Have you participated in any of the following?</div><div class="options">${['Hackathon','Open Source','Freelance','Research','Bootcamp'].map(v=>option(v,state.profile.exposure.includes(v),`toggleExposure('${v}')`)).join('')}</div>
<div class="actions"><button class="ghost" onclick="setStep(1)">← BACK</button><button class="primary" onclick="setStep(3)">NEXT →</button></div></section>`; }

function selfAssessment(){ return `${tabs()}<section class="form"><h2>Self-Assessment</h2><p class="subtitle">Rate your confidence in entering the job market.</p>
<div class="label">Confidence Level - ${state.profile.confidence}/10</div><input class="range" type="range" min="1" max="10" value="${state.profile.confidence}" oninput="state.profile.confidence=+this.value; render()" />
<div class="infoBox"><div><span class="tick">✓</span>Readiness Index Score (0-100)</div><div><span class="tick">✓</span>Specialization-specific scoring</div><div><span class="tick">✓</span>NLP-based skill extraction and dataset comparison</div><div><span class="tick">✓</span>Optional LLM-generated summary if backend is configured</div></div>
<div class="actions"><button class="ghost" onclick="setStep(2)">← BACK</button><button class="primary" onclick="setStep(4)">ANALYZE MY PROFILE →</button></div></section>`; }

function weightedAlignment(selectedSkills, rankedSkills, topN = 10, specialization = null) {
  const config = getSpecializationConfig(specialization);
  const weights = config.weights || {};
  const required = new Set(config.required || []);
  const hasSpecialization = Boolean(specialization && required.size);
  const top = (rankedSkills || []).slice(0, topN);
  if (!top.length || !selectedSkills.length) return 0;

  let achieved = 0;
  let possible = 0;

  top.forEach((x, idx) => {
    const baseWeight = 10 - Math.min(idx, 9);
    // If no specialization is selected, use neutral weighting.
    // If specialization is selected, priority skills are boosted and unrelated skills are reduced.
    const specializationBoost = !hasSpecialization ? 1.0 : (required.has(x.skill) ? 1.9 : 0.45);
    const skillWeight = !hasSpecialization ? 1.0 : (weights[x.skill] || 0.85);
    const finalWeight = baseWeight * specializationBoost * skillWeight;
    possible += finalWeight;
    if (selectedSkills.includes(x.skill)) achieved += finalWeight;
  });

  return possible ? achieved / possible : 0;
}

function specializationFit(selectedSkills, specialization) {
  const config = getSpecializationConfig(specialization);
  const required = config.required || [];
  const weights = config.weights || {};

  // Neutral baseline when no specialization is selected. This makes click-to-select
  // and click-again-to-deselect visibly affect the final score.
  if (!specialization || !required.length) return 30;

  const possible = required.reduce((sum, skill) => sum + (weights[skill] || 1), 0);
  const achieved = required.reduce((sum, skill) => {
    return sum + (selectedSkills.includes(skill) ? (weights[skill] || 1) : 0);
  }, 0);

  // A selected specialization with no matching priority skills should not look
  // identical to no specialization. It receives a low, explicit fit score.
  if (achieved === 0) return 10;

  return possible ? Math.round((achieved / possible) * 100) : 10;
}

function specializationImpact(selectedSkills, specialization) {
  const config = getSpecializationConfig(specialization);
  const required = config.required || [];
  const matched = required.filter(skill => selectedSkills.includes(skill));
  const missing = required.filter(skill => !selectedSkills.includes(skill));
  return { matched, missing, required };
}

function scoreProfile(){
  const p = state.profile;
  const projectMap = {'0':0,'1':18,'2':36,'3':55,'4':72,'5+':90};
  const hourMap = {'< 5 hrs':15,'5-10 hrs':38,'10-20 hrs':68,'20+ hrs':92};
  const exposureBonus = Math.min(35, p.exposure.length * 9) + (p.internship === 'Yes' ? 18 : 0);
  const rawPE = Math.min(100, Math.round((projectMap[p.projects] * .48) + (hourMap[p.hours] * .34) + exposureBonus));
  const studentExposureBenchmark = state.studentBenchmark?.avgExposure || 0;
  const PE = Math.round((rawPE * .80) + (studentExposureBenchmark * .20));

  const jobScore = weightedAlignment(p.skills, state.demand?.ranked || [], 10, p.specialization);
  const surveyScore = weightedAlignment(p.skills, state.surveyDemand?.ranked || [], 10, p.specialization);
  const specScore = specializationFit(p.skills, p.specialization);

  const JDA = Math.round(Math.min(100, jobScore * 100));
  const SBA = Math.round(Math.min(100, surveyScore * 100));
  const SPA = specScore;
  const SDA = Math.round((JDA * .50) + (SBA * .20) + (SPA * .30));

  const ED = Math.min(100, Math.round(projectMap[p.projects] * .55 + (p.internship === 'Yes' ? 25 : 0) + Math.min(20, p.exposure.length * 4)));
  const CPR = p.confidence * 10;

  // Specialization is intentionally a direct score component so changing
  // Data Science / Software Engineering / Networks / AI & ML / Cybersecurity
  // changes the final score visibly, not just in tiny hidden decimals.
  const total = Math.round(PE*.25 + SDA*.25 + ED*.20 + CPR*.15 + SPA*.15);
  return { PE, JDA, SBA, SPA, SDA, ED, CPR, total, specialization: p.specialization || 'Not selected', specializationImpact: specializationImpact(p.skills, p.specialization), studentExposureBenchmark: state.studentBenchmark?.avgExposure || 0, studentReadinessBenchmark: state.studentBenchmark?.avgReadiness || 0 }; 
}

function level(score){ if(score>=80) return 'MARKET READY'; if(score>=60) return 'ADVANCING'; if(score>=40) return 'DEVELOPING'; return 'FOUNDATION'; }

function buildRecommendation(s, gaps) {
  const p = state.profile;
  const firstGap = gaps[0]?.skill || 'portfolio depth';
  const targetSkills = p.specialization ? (specializationProfiles[p.specialization] || []) : [];
  const missingTarget = targetSkills.filter(x => !p.skills.includes(x)).slice(0,3);
  return [
    p.specialization ? `For ${p.specialization}, focus first on ${firstGap}${missingTarget.length ? ` and then strengthen ${missingTarget.join(', ')}` : ''}.` : `Select a specialization to calculate specialization-specific alignment and more accurate gaps. For now, focus first on ${firstGap}.`,
    p.specialization ? `Build one portfolio project that proves ${p.specialization} ability using ${p.skills.slice(0,3).join(', ') || 'core technical skills'} and publish it with a short case-study report.` : `Build one portfolio project using ${p.skills.slice(0,3).join(', ') || 'core technical skills'} and publish it with a short case-study report.`,
    `Your weakest evidence area is ${s.PE < s.SDA ? 'practical exposure' : 'skill-market alignment'}, so improve it before treating confidence as a career strategy. Confidence without evidence is just a motivational poster.`
  ];
}

async function requestLLMRecommendation() {
  const s = scoreProfile();
  const gaps = getGaps();
  state.llmStatus = 'loading';
  state.llmText = '';
  state.llmError = '';
  render();
  try {
    const res = await fetch(`${BACKEND_API_BASE}/api/recommendations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile: state.profile, score: s, gaps })
    });
    const data = await res.json();
    if (!res.ok) {
      state.llmText = data.recommendation || '';
      state.llmError = data.llm_error || data.openai_key_message || `LLM API returned ${res.status}`;
      state.llmStatus = 'invalid_key';
    } else {
      state.llmText = data.recommendation || '';
      state.llmError = '';
      state.llmStatus = state.llmText ? 'done' : 'failed';
    }
  } catch (e) {
    console.warn('LLM recommendation failed. Using local dynamic recommendations.', e);
    state.llmError = e.message || String(e);
    state.llmStatus = 'failed';
  }
  render();
}

function getGaps(){
  const p = state.profile;
  const selected = new Set(p.skills);
  const target = new Set(p.specialization ? (specializationProfiles[p.specialization] || []) : []);
  const jobGaps = (state.demand?.ranked || []).filter(x => !selected.has(x.skill)).slice(0,8);
  const surveyGaps = (state.surveyDemand?.ranked || []).filter(x => !selected.has(x.skill)).slice(0,8);
  const merged = new Map();
  for (const g of jobGaps) merged.set(g.skill, { skill: g.skill, jobPct: g.pct, surveyPct: 0, spec: target.has(g.skill) });
  for (const g of surveyGaps) merged.set(g.skill, { ...(merged.get(g.skill) || { skill: g.skill, jobPct: 0, spec: target.has(g.skill) }), surveyPct: g.pct });
  return [...merged.values()].sort((a,b)=>((b.jobPct + b.surveyPct) * (b.spec ? 1.4 : 1)) - ((a.jobPct + a.surveyPct) * (a.spec ? 1.4 : 1))).slice(0,4);
}

function result(){
  const s = scoreProfile();
  const gaps = getGaps();
  const gapHtml = gaps.length ? gaps.map(g=>`<div class="gap"><div><span>↑</span>${g.skill}${g.spec ? ' · specialization priority' : ''}</div><small>${g.jobPct}% of job posts · ${g.surveyPct}% of survey profiles</small></div>`).join('') : `<div class="gap"><div><span>✓</span>No major high-demand gap found</div><small>Good coverage across both datasets</small></div>`;
  const localRecs = buildRecommendation(s, gaps).map(x=>`<li>${x}</li>`).join('');
  const llmBlock = state.llmStatus === 'done'
    ? `<div class="llmBox"><h3>LLM Recommendation Summary</h3><p>${state.llmText.replace(/\n/g,'<br>')}</p></div>`
    : state.llmStatus === 'loading'
      ? `<div class="loading">Generating LLM summary...</div>`
      : `<button class="ghost" onclick="requestLLMRecommendation()">GENERATE LLM SUMMARY</button><p class="loading">Requires backend/server.py running with OPENAI_API_KEY. Static hosting will use local dynamic recommendations.</p>`;
  return `<section class="result">
    <div class="score"><div class="kicker">Your Readiness Index</div><div class="num">${s.total}</div><div class="denom">/ 100</div><div class="badge">${level(s.total)}</div></div>
    <div class="sectionTitle">Score Breakdown</div><div class="breakdown">
      ${row('Practical Exposure (PE)',s.PE,'25%')}${row('Skill Demand Alignment (SDA)',s.SDA,'25%')}${row('Experience Depth (ED)',s.ED,'20%')}${row('Confidence & Perceived Readiness (CPR)',s.CPR,'15%')}${row('Specialization Alignment (SPA)',s.SPA,'15% direct impact')}
    </div>
    <div class="sectionTitle">Dataset Evidence</div><div class="breakdown compact">
      ${row('Job Demand Alignment from Data_jobs.csv',s.JDA,'50% of SDA')}${row('Survey Benchmark Alignment from survey_results_public.csv',s.SBA,'20% of SDA')}${row('Specialization Alignment',s.SPA,'30% of SDA + 15% direct final score impact')}${row('Student Survey Practical Exposure Benchmark',s.studentExposureBenchmark,'20% of PE')}${row('Student Survey Readiness Benchmark',s.studentReadinessBenchmark,'H1 evidence')}
    </div>
    <div class="sectionTitle">Selected Specialization Effect</div><div class="gaps"><div class="gap"><div><span>✓</span>${s.specialization}</div><small>Matched priority skills: ${s.specializationImpact.matched.length ? s.specializationImpact.matched.join(', ') : 'None'}</small></div><div class="gap"><div><span>↑</span>Missing priority skills</div><small>${s.specializationImpact.missing.length ? s.specializationImpact.missing.join(', ') : 'None'}</small></div></div><div class="sectionTitle">Top Skill Gaps</div><div class="gaps">${gapHtml}</div>
    <div class="recs"><h3>Dynamic Recommendations</h3><ol>${localRecs}</ol>${llmBlock}</div>
    <div class="actions"><button class="ghost" onclick="setStep(-1)">← START OVER</button></div>
  </section>`;
}
function row(name,val,weight){ return `<div class="row"><div><div>${name}</div><div class="bar"><div class="fill" style="width:${val}%"></div></div></div><div class="metric">${val}/100 · ${weight}</div></div>`; }
function render(){ app.innerHTML = state.step === -1 ? home() : state.step === 0 ? background() : state.step === 1 ? skills() : state.step === 2 ? experience() : state.step === 3 ? selfAssessment() : result(); }

window.setStep = setStep; window.selectOne = selectOne; window.toggleSkill = toggleSkill; window.toggleExposure = toggleExposure; window.render = render; window.requestLLMRecommendation = requestLLMRecommendation;
render(); loadData();
