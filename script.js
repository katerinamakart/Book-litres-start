// ── Падающие капельки дождя ──
const rainContainer = document.getElementById('rain');

for (let i = 0; i < 70; i++) {
  const drop = document.createElement('div');
  drop.className = 'rain-drop';

  const height = 14 + Math.random() * 18;   // длина капли 14–32px
  drop.style.left             = Math.random() * 100 + '%';
  drop.style.height           = height + 'px';
  drop.style.animationDuration = (0.6 + Math.random() * 1.4) + 's';
  drop.style.animationDelay   = (Math.random() * 4) + 's';
  drop.style.opacity          = 0.25 + Math.random() * 0.5;

  rainContainer.appendChild(drop);
}

// ── Плавное появление блоков при прокрутке ──
const faders = document.querySelectorAll('.fade');

const appear = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('show');
    }
  });
}, { threshold: 0.12, rootMargin: '0px 0px -30px 0px' });

faders.forEach(el => appear.observe(el));

// ── «Живая» кнопка: небольшое покачивание при наведении ──
document.querySelectorAll('.button').forEach(btn => {
  btn.addEventListener('mouseenter', () => {
    btn.style.animationPlayState = 'paused';
  });
  btn.addEventListener('mouseleave', () => {
    btn.style.animationPlayState = 'running';
  });
});

// ── Плавный переход по якорным ссылкам ──
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});
