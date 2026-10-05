// Premium UI Interactions - Smooth & Clear
document.addEventListener('DOMContentLoaded', function() {
    initPageLoadAnimation();
    initSmoothScrolling();
    initCardHoverEffects();
    initFormInteractions();
    initNavbarEffects();
});

// Page Load Animation
function initPageLoadAnimation() {
    const elements = document.querySelectorAll('h1, h2, h3, .feature-card, .step-card, .stat-card');

    elements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        el.style.transitionDelay = (index * 0.05) + 's';

        setTimeout(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, 100);
    });
}

// Smooth Scrolling
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href && href !== '#') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    });
}

// Card Hover Effects
function initCardHoverEffects() {
    const cards = document.querySelectorAll('.feature-card, .step-card, .stat-card');

    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

// Form Interactions
function initFormInteractions() {
    const inputs = document.querySelectorAll('.form-control, .form-select');

    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.style.borderColor = 'var(--primary)';
            this.style.boxShadow = '0 0 0 3px rgba(91, 91, 255, 0.1)';
        });

        input.addEventListener('blur', function() {
            this.style.borderColor = 'var(--border-color)';
            this.style.boxShadow = 'none';
        });
    });
}

// Navbar Effects
function initNavbarEffects() {
    const navbar = document.querySelector('.navbar');

    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.08)';
        }
    });
}

// Button Ripple Effect
document.querySelectorAll('.btn-hero-primary, .btn-analyze, .btn-cta, .btn-primary-nav').forEach(button => {
    button.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;

        ripple.style.position = 'absolute';
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.style.borderRadius = '50%';
        ripple.style.background = 'rgba(255, 255, 255, 0.6)';
        ripple.style.pointerEvents = 'none';
        ripple.style.animation = 'rippleAnimation 0.6s ease-out';

        if (this.position === 'static') {
            this.style.position = 'relative';
        }

        this.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes rippleAnimation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Floating Label Animation
document.querySelectorAll('.form-control').forEach(input => {
    if (input.value) {
        const label = input.previousElementSibling;
        if (label) label.style.transform = 'scale(0.9) translateY(-10px)';
    }

    input.addEventListener('focus', function() {
        const label = this.previousElementSibling;
        if (label && label.classList.contains('form-label')) {
            label.style.transform = 'scale(0.9) translateY(-10px)';
            label.style.color = 'var(--primary)';
        }
    });

    input.addEventListener('blur', function() {
        if (!this.value) {
            const label = this.previousElementSibling;
            if (label) {
                label.style.transform = 'scale(1) translateY(0)';
                label.style.color = 'var(--text-secondary)';
            }
        }
    });
});

// Counter Animation
function animateCounter(element, target, duration = 2000) {
    let current = 0;
    const increment = target / (duration / 16);

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

// Intersect Observer for Counter Animation
const statNumbers = document.querySelectorAll('.stat-number');
const counterObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.dataset.animated) {
            const value = parseInt(entry.target.textContent);
            if (!isNaN(value)) {
                animateCounter(entry.target, value);
                entry.target.dataset.animated = true;
            }
        }
    });
}, { threshold: 0.5 });

statNumbers.forEach(el => counterObserver.observe(el));

// Loading State
function showLoadingState(button) {
    const originalContent = button.innerHTML;
    button.innerHTML = '<span style="display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,0.3);border-top-color:white;border-radius:50%;animation:spin 0.8s linear infinite;"></span> Processing...';
    button.disabled = true;
    return originalContent;
}

function hideLoadingState(button, originalContent) {
    button.innerHTML = originalContent;
    button.disabled = false;
}

// Parallax Effect
window.addEventListener('scroll', function() {
    const parallaxElements = document.querySelectorAll('.hero-bg');
    parallaxElements.forEach(el => {
        el.style.transform = `translateY(${window.scrollY * 0.3}px)`;
    });
});

// Intersection Observer for Fade-in
const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.feature-card, .step-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});
