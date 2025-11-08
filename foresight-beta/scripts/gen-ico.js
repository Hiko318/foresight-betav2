const fs = require('fs');
const path = require('path');
let pngToIco = require('png-to-ico');
// Handle ESM default export shape
pngToIco = pngToIco && pngToIco.default ? pngToIco.default : pngToIco;
const sharp = require('sharp');

(async () => {
  const src = path.join(__dirname, '..', 'assets', 'icon.png');
  const out = path.join(__dirname, '..', 'assets', 'icon.ico');
  const tmp = path.join(__dirname, '..', 'assets', 'icon-square.png');

  if (!fs.existsSync(src)) {
    console.error('Missing assets/icon.png');
    process.exit(1);
  }

  try {
    // Ensure the PNG is square; pad transparent if needed using sharp
    const meta = await sharp(src).metadata();
    const w = meta.width || 0;
    const h = meta.height || 0;
    let inputForIco = src;

    if (w !== h) {
      const size = Math.max(w, h);
      await sharp(src)
        .resize({ width: size, height: size, fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
        .toFile(tmp);
      inputForIco = tmp;
      console.log(`Padded non-square PNG (${w}x${h}) to ${size}x${size}`);
    }

    const buf = await pngToIco(inputForIco);
    fs.writeFileSync(out, buf);
    console.log('Wrote', out);

    if (fs.existsSync(tmp)) {
      try { fs.unlinkSync(tmp); } catch {}
    }
  } catch (err) {
    console.error('Failed to generate ICO:', err.message);
    process.exit(1);
  }
})();