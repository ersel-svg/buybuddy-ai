module.exports = {
  ci: {
    collect: {
      // Start server before running tests
      startServerCommand: 'npm run start',
      startServerReadyPattern: 'Ready in',
      startServerReadyTimeout: 30000,

      // URLs to test - grouped by priority
      url: [
        // Core pages (most used)
        'http://localhost:3000/',
        'http://localhost:3000/products',
        'http://localhost:3000/datasets',
        'http://localhost:3000/workflows',

        // Data management pages
        'http://localhost:3000/videos',
        'http://localhost:3000/cutouts',
        'http://localhost:3000/triplets',
        'http://localhost:3000/augmentation',
        'http://localhost:3000/embeddings',

        // Object Detection pages
        'http://localhost:3000/od',
        'http://localhost:3000/od/datasets',
        'http://localhost:3000/od/images',
        'http://localhost:3000/od/training',
        'http://localhost:3000/od/annotate',

        // Classification pages
        'http://localhost:3000/classification',
        'http://localhost:3000/classification/datasets',
        'http://localhost:3000/classification/images',
        'http://localhost:3000/classification/training',
        'http://localhost:3000/classification/labeling',

        // Other pages
        'http://localhost:3000/training',
        'http://localhost:3000/matching',
        'http://localhost:3000/scan-requests',
        'http://localhost:3000/workflows/executions',
        'http://localhost:3000/workflows/models',
        'http://localhost:3000/products/matcher',
        'http://localhost:3000/products/bulk-update',

        // Auth
        'http://localhost:3000/login',
      ],

      // Number of runs per URL for more accurate results
      numberOfRuns: 3,

      // Lighthouse settings
      settings: {
        preset: 'desktop',
        // Throttling for realistic conditions
        throttling: {
          rttMs: 40,
          throughputKbps: 10240,
          cpuSlowdownMultiplier: 1,
        },
        // Skip some audits for faster runs
        skipAudits: ['uses-http2'],
        // Chrome flags
        chromeFlags: ['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage'],
      },
    },

    assert: {
      // Performance budgets
      assertions: {
        // Core Web Vitals
        'first-contentful-paint': ['warn', { maxNumericValue: 2000 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 4000 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'total-blocking-time': ['warn', { maxNumericValue: 300 }],
        'speed-index': ['warn', { maxNumericValue: 4000 }],

        // Performance score
        'categories:performance': ['warn', { minScore: 0.7 }],
        'categories:accessibility': ['warn', { minScore: 0.8 }],
        'categories:best-practices': ['warn', { minScore: 0.8 }],
        'categories:seo': ['warn', { minScore: 0.7 }],

        // Resource budgets
        'resource-summary:script:size': ['warn', { maxNumericValue: 500000 }], // 500KB JS
        'resource-summary:stylesheet:size': ['warn', { maxNumericValue: 100000 }], // 100KB CSS
        'resource-summary:image:size': ['warn', { maxNumericValue: 500000 }], // 500KB images
        'resource-summary:total:size': ['warn', { maxNumericValue: 2000000 }], // 2MB total

        // Network requests
        'network-requests': ['warn', { maxNumericValue: 50 }],

        // Specific audits
        'uses-long-cache-ttl': 'warn',
        'uses-text-compression': 'warn',
        'uses-responsive-images': 'warn',
        'efficient-animated-content': 'warn',
        'duplicated-javascript': 'warn',
        'legacy-javascript': 'warn',
        'dom-size': ['warn', { maxNumericValue: 1500 }],
        'mainthread-work-breakdown': ['warn', { maxNumericValue: 4000 }],
        'bootup-time': ['warn', { maxNumericValue: 3000 }],
        'unused-javascript': 'warn',
        'unused-css-rules': 'warn',
        'render-blocking-resources': 'warn',
        'unminified-javascript': 'error',
        'unminified-css': 'error',
      },
    },

    upload: {
      // Save reports locally
      target: 'filesystem',
      outputDir: './lighthouse-reports',
      reportFilenamePattern: '%%PATHNAME%%-%%DATETIME%%-report.%%EXTENSION%%',
    },
  },
};
