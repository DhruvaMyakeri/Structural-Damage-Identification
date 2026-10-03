const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pres.layout = "WIDE";
pres.author = "Dhruva Myakeri and Navya Suresh";
pres.title = "Structural Damage Image Classification";
pres.subject = "UE24CS352A Team 16 Project 31";

const INK = "1F2428";
const COPPER = "C45C26";
const TEAL = "1F4E5F";
const PAPER = "FFFFFF";
const CARD = "F3F5F6";
const MUTED = "5C656B";
const WHITE = "FFFFFF";

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color: fill || CARD },
    rectRadius: 0.08,
  });
}

function addText(slide, text, opts) {
  slide.addText(text, Object.assign({ isTextBox: true, fontFace: "Calibri", color: INK, margin: 0 }, opts));
}

// 1. Title
{
  const s = pres.addSlide();
  s.background = { color: INK };
  addText(s, "Structural damage,\nfrom a photograph", {
    x: 0.55, y: 1.35, w: 8.2, h: 2.2,
    fontFace: "Cambria", fontSize: 40, bold: true, color: WHITE, margin: 0,
  });
  addText(s, "UE24CS352A  ·  Section C  ·  Team 16  ·  Project 31", {
    x: 0.55, y: 3.75, w: 8, h: 0.35,
    fontSize: 16, color: "D7D3CC", margin: 0,
  });
  addText(s, "Dhruva Myakeri  (PES1UG24CS156)\nNavya Suresh  (PES1UG24CS904)", {
    x: 0.55, y: 4.3, w: 7, h: 0.8,
    fontSize: 18, color: WHITE, margin: 0,
  });
  addText(s, "Following Ho and Troncoso, CS229 2018", {
    x: 0.55, y: 6.55, w: 7, h: 0.35,
    fontSize: 14, color: "A8B0B6", margin: 0,
  });
  card(s, 9.15, 1.7, 3.55, 3.6, "2A3136");
  addText(s, "99.70%", {
    x: 9.35, y: 2.15, w: 3.15, h: 1.1,
    fontFace: "Cambria", fontSize: 42, bold: true, color: "E7B89A", align: "center", margin: 0,
  });
  addText(s, "test accuracy\nbest network\non all 8,000\ntest photos", {
    x: 9.4, y: 3.35, w: 3.05, h: 1.5,
    fontSize: 16, color: WHITE, align: "center", margin: 0,
  });
  s.addNotes("Open with the task, not the architecture. A photo in, damaged or undamaged out. The 99.70% is MobileNetV2 with the last blocks unfrozen, on the full test split. Do not call it 100%. Chance on this balanced set is 50%.");
}

// 2. Problem
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  addText(s, "The decision", {
    x: 0.5, y: 0.38, w: 8, h: 0.55,
    fontFace: "Cambria", fontSize: 32, bold: true, margin: 0,
  });
  addText(s, "After an earthquake, which structures are damaged, without walking every street.", {
    x: 0.5, y: 1.05, w: 12, h: 0.4,
    fontSize: 16, color: MUTED, margin: 0,
  });
  card(s, 0.5, 1.75, 6.0, 4.7, "F7F1EC");
  addText(s, "Class 1", {
    x: 0.75, y: 2.0, w: 5.5, h: 0.35,
    fontSize: 14, color: COPPER, bold: true, margin: 0,
  });
  addText(s, "Damaged", {
    x: 0.75, y: 2.4, w: 5.5, h: 0.6,
    fontFace: "Cambria", fontSize: 32, bold: true, margin: 0,
  });
  addText(s, "In the paper: cracks through to a levelled building, plus clutter, blur, and some wrong labels.\n\nIn this run: a close-up of concrete with a crack. Same label, narrower picture.", {
    x: 0.75, y: 3.2, w: 5.4, h: 2.6,
    fontSize: 16, margin: 0,
  });
  card(s, 6.8, 1.75, 6.0, 4.7, "E7F0F2");
  addText(s, "Class 0", {
    x: 7.05, y: 2.0, w: 5.5, h: 0.35,
    fontSize: 14, color: TEAL, bold: true, margin: 0,
  });
  addText(s, "Undamaged", {
    x: 7.05, y: 2.4, w: 5.5, h: 0.6,
    fontFace: "Cambria", fontSize: 32, bold: true, margin: 0,
  });
  addText(s, "In the paper: about 54% of 5,913 photos, so a lazy guess of undamaged scores about 54%.\n\nIn this run: exactly half of every split. A lazy guess scores 50%. Anything we report has to clear that.", {
    x: 7.05, y: 3.2, w: 5.4, h: 2.6,
    fontSize: 16, margin: 0,
  });
  s.addNotes("Define the label before any model name. Class 0 undamaged, class 1 damaged, matching the paper. Stress that our photos are easier than PEER's.");
}

// 3. Paper vs us
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  addText(s, "Same question, different photos", {
    x: 0.5, y: 0.35, w: 12, h: 0.5,
    fontFace: "Cambria", fontSize: 32, bold: true, margin: 0,
  });
  const rows = [
    [
      { text: "", options: { fill: { color: TEAL }, color: WHITE, bold: true } },
      { text: "Ho and Troncoso, 2018", options: { fill: { color: TEAL }, color: WHITE, bold: true } },
      { text: "This run", options: { fill: { color: TEAL }, color: WHITE, bold: true } },
    ],
    ["Photos", "PEER Hub ImageNet, 5,913", "Concrete crack close-ups, 40,000"],
    ["What 'damaged' looks like", "Crack, spall, or collapse, plus clutter", "A crack on otherwise plain concrete"],
    ["Split", "90% train, 10% called both val and test", "Official train, validation, and test"],
    ["Flat-vector models", "Validation near 51% to 55%", "Test 75% to 93% on 32 by 32 pixels"],
    ["Best reported network", "InceptionV3, 83% validation", "MobileNetV2 unfrozen, 99.70% test"],
  ];
  s.addTable(rows, {
    x: 0.5, y: 1.15, w: 12.3, h: 5.2,
    colW: [3.1, 4.6, 4.6],
    border: [{ pt: 0, color: PAPER }, { pt: 0, color: PAPER }, { pt: 0, color: PAPER }, { pt: 0, color: PAPER }],
    fontFace: "Calibri",
    fontSize: 15,
    color: INK,
    valign: "middle",
    align: "left",
  });
  s.addNotes("If asked why we did not use PEER: the files are emailed after a form. We say that in the write-up. The interesting result is that the ranking changes because the photos got easier, not because we invented a better network.");
}

// 4. Counts
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  addText(s, "How many photos, and which ones the models saw", {
    x: 0.5, y: 0.35, w: 12, h: 0.5,
    fontFace: "Cambria", fontSize: 28, bold: true, margin: 0,
  });
  const blocks = [
    ["12,000 + 12,000", "Training archive\nundamaged + damaged"],
    ["4,000 + 4,000", "Validation archive\nused to pick the demo model"],
    ["4,000 + 4,000", "Test archive\nnot used to pick anything"],
  ];
  blocks.forEach((item, i) => {
    const x = 0.5 + i * 4.2;
    card(s, x, 1.2, 3.95, 2.5);
    addText(s, item[0], {
      x: x + 0.2, y: 1.4, w: 3.55, h: 0.7,
      fontFace: "Cambria", fontSize: 22, bold: true, color: TEAL, margin: 0,
    });
    addText(s, item[1], {
      x: x + 0.2, y: 2.2, w: 3.55, h: 1.1,
      fontSize: 16, margin: 0,
    });
  });
  addText(s, "Not every model sees the whole archive. Flat-pixel models train on 1,500 per class at 32 by 32, because a full-resolution vector is 150,528 numbers and an RBF kernel on that is not practical. Networks train on 3,000 per class and are scored on all 8,000 validation and 8,000 test photos. The feature SVM is scored on 2,400 test photos. Quote the count with the percentage.", {
    x: 0.5, y: 4.05, w: 12.3, h: 1.5,
    fontSize: 16, margin: 0,
  });
  addText(s, "Pixels for the flat models are mapped with (pixel / 128) - 1, the paper's formula. Network inputs use ImageNet mean and standard deviation, because that is what the downloaded weights expect.", {
    x: 0.5, y: 5.6, w: 12.3, h: 1.2,
    fontSize: 16, margin: 0,
  });
  s.addNotes("If someone asks why 32 by 32: say the number 150,528 and that SVM training builds a matrix of every training pair. We did not silently change the formula. We changed the resolution and wrote it down.");
}

// 5. Pipeline
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  addText(s, "Eight models, one comparison", {
    x: 0.5, y: 0.32, w: 12, h: 0.5,
    fontFace: "Cambria", fontSize: 32, bold: true, margin: 0,
  });
  const steps = [
    ["1", "Flat pixels", "k = 5 neighbours, L2 logistic regression with C = 1, RBF SVM with gamma = 0.001. Same constants as the paper."],
    ["2", "Frozen trunk", "MobileNetV2 and InceptionV3 keep ImageNet weights. Only a new 2-class layer is trained. Cross-entropy is their softmax."],
    ["3", "More layers", "Last four MobileNet blocks unfrozen, continued from the frozen checkpoint, smaller learning rate."],
    ["4", "Augment", "A fresh MobileNet sees horizontal flips and a small shift. We did not zoom. The paper showed zoom can invent invalid pictures."],
    ["5", "Mid-layer SVM", "Inception block Mixed_6e, averaged to 768 numbers, then an RBF SVM. Gamma picked on validation."],
  ];
  steps.forEach((step, i) => {
    const y = 1.05 + i * 1.15;
    card(s, 0.5, y, 0.7, 0.7, TEAL);
    addText(s, step[0], {
      x: 0.5, y: y + 0.12, w: 0.7, h: 0.45,
      fontSize: 20, bold: true, color: WHITE, align: "center", margin: 0,
    });
    addText(s, step[1], {
      x: 1.45, y: y, w: 3.2, h: 0.7,
      fontSize: 18, bold: true, margin: 0,
    });
    addText(s, step[2], {
      x: 4.7, y: y, w: 8.1, h: 0.95,
      fontSize: 15, margin: 0,
    });
  });
  s.addNotes("Walk this in order. The unfrozen run is not a new network from scratch. It starts from the frozen weights. Say that, because a random head at learning rate 0.0001 would not have learned.");
}

// 6. Chart
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  addText(s, "Test accuracy", {
    x: 0.5, y: 0.28, w: 8, h: 0.45,
    fontFace: "Cambria", fontSize: 32, bold: true, margin: 0,
  });
  addText(s, "Chance on this set is 50%. The feature SVM bar is on 2,400 photos. The others are on 4,000 or 8,000.", {
    x: 0.5, y: 0.8, w: 12, h: 0.35,
    fontSize: 14, color: MUTED, margin: 0,
  });
  s.addChart(pres.charts.BAR, [
    {
      name: "Test accuracy (%)",
      labels: ["KNN", "Logistic", "SVM pixels", "MobileNet\nfrozen", "MobileNet\nunfrozen", "MobileNet\n+ aug", "InceptionV3", "SVM on\nfeatures"],
      values: [75.12, 87.9, 92.97, 99.45, 99.7, 99.03, 98.46, 99.96],
    },
  ], {
    x: 0.4, y: 1.2, w: 12.5, h: 5.8,
    barGrouping: "clustered",
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelFontSize: 11,
    dataLabelColor: INK,
    dataLabelFontFace: "Calibri",
    showLegend: false,
    showTitle: false,
    chartColors: [COPPER],
    catAxisLabelColor: INK,
    valAxisLabelColor: INK,
    catAxisLabelFontSize: 11,
    valAxisLabelFontSize: 11,
    valAxisMinValue: 0,
    valAxisMaxValue: 118,
    valGridLine: { color: "E2E6E8", size: 0.5 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: PAPER } },
  });
  s.addNotes("Read left to right. KNN is the weak flat model. Logistic and the pixel SVM already beat the paper's classical results by a lot. The four networks sit on top of each other. Do not rank Inception above MobileNet. The numbers go the other way.");
}

// 7. KNN surprise
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  addText(s, "KNN is 75% accurate and still misses half the cracks", {
    x: 0.5, y: 0.32, w: 12.3, h: 0.55,
    fontFace: "Cambria", fontSize: 28, bold: true, margin: 0,
  });
  card(s, 0.5, 1.15, 4.0, 2.3);
  addText(s, "100%", {
    x: 0.7, y: 1.35, w: 3.6, h: 0.9,
    fontFace: "Cambria", fontSize: 40, bold: true, color: TEAL, margin: 0,
  });
  addText(s, "of its 'damaged' calls\nwere actually damaged", {
    x: 0.7, y: 2.3, w: 3.6, h: 0.8,
    fontSize: 16, margin: 0,
  });
  card(s, 4.7, 1.15, 4.0, 2.3);
  addText(s, "50.25%", {
    x: 4.9, y: 1.35, w: 3.6, h: 0.9,
    fontFace: "Cambria", fontSize: 40, bold: true, color: COPPER, margin: 0,
  });
  addText(s, "of real cracks\nwere found", {
    x: 4.9, y: 2.3, w: 3.6, h: 0.8,
    fontSize: 16, margin: 0,
  });
  card(s, 8.9, 1.15, 3.9, 2.3);
  addText(s, "0", {
    x: 9.1, y: 1.35, w: 3.5, h: 0.9,
    fontFace: "Cambria", fontSize: 40, bold: true, color: INK, margin: 0,
  });
  addText(s, "intact photos\ncalled damaged", {
    x: 9.1, y: 2.3, w: 3.5, h: 0.8,
    fontSize: 16, margin: 0,
  });
  addText(s, "Test counts on 4,000 photos: 2,000 true undamaged, 0 false damaged, 995 missed cracks, 1,005 found cracks. A crack is a thin line. Flatten the photo and that line is just a few coordinates. Move the crack and those coordinates change, so the five nearest training vectors are often intact concrete. Accuracy alone hides this. Precision and recall do not.", {
    x: 0.5, y: 3.7, w: 12.3, h: 1.6,
    fontSize: 16, margin: 0,
  });
  addText(s, "Logistic regression's validation accuracy climbs from 76.33% at 300 training photos to 82.03% at 1,500, then sits there. More of the same flat vectors will not reach the network.", {
    x: 0.5, y: 5.45, w: 12.3, h: 1.3,
    fontSize: 16, margin: 0,
  });
  s.addNotes("This is the slide to slow down on. Accuracy 75% sounds mediocre. The confusion matrix says the model almost refuses to say damaged. That is a property of nearest neighbours on a translated thin pattern.");
}

// 8. Networks
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  addText(s, "What extra training did, and did not do", {
    x: 0.5, y: 0.32, w: 12, h: 0.5,
    fontFace: "Cambria", fontSize: 30, bold: true, margin: 0,
  });
  const nets = [
    ["99.45%", "Frozen MobileNet", "2,562 trained numbers\nout of 2,226,434"],
    ["99.70%", "Last blocks unfrozen", "1,528,642 trained numbers.\nSmall gain, train still 99.97%"],
    ["99.03%", "Flip and shift", "Slightly worse than the\nfrozen run. Augmentation\ndid not buy accuracy."],
    ["98.46%", "Frozen InceptionV3", "The paper's winner.\nHere it is behind MobileNet."],
  ];
  nets.forEach((item, i) => {
    const x = 0.45 + (i % 4) * 3.2;
    card(s, x, 1.15, 3.05, 3.15);
    addText(s, item[0], {
      x: x + 0.15, y: 1.35, w: 2.75, h: 0.7,
      fontFace: "Cambria", fontSize: 28, bold: true, color: i === 2 ? COPPER : TEAL, margin: 0,
    });
    addText(s, item[1], {
      x: x + 0.15, y: 2.15, w: 2.75, h: 0.7,
      fontSize: 15, bold: true, margin: 0,
    });
    addText(s, item[2], {
      x: x + 0.15, y: 2.9, w: 2.75, h: 1.15,
      fontSize: 13, color: MUTED, margin: 0,
    });
  });
  addText(s, "The feature SVM reaches 99.96% on 2,400 test photos, with one false damaged call and no missed crack. Gamma 0.03 was the best value in the grid, and validation was still rising, so we have not shown it is the peak.", {
    x: 0.5, y: 4.55, w: 12.3, h: 1.1,
    fontSize: 16, margin: 0,
  });
  addText(s, "Demo model, chosen only by validation accuracy: that SVM. python demo.py", {
    x: 0.5, y: 5.8, w: 12.3, h: 0.8,
    fontSize: 18, bold: true, color: TEAL, margin: 0,
  });
  s.addNotes("Unfreezing helped by a quarter of a point. That is real and small. Augmentation hurt by about four tenths of a point. Do not claim the paper's bias story showed up. On this data the frozen network was already near the ceiling.");
}

// 9. Close
{
  const s = pres.addSlide();
  s.background = { color: INK };
  addText(s, "What we can say", {
    x: 0.55, y: 0.4, w: 12, h: 0.55,
    fontFace: "Cambria", fontSize: 32, bold: true, color: WHITE, margin: 0,
  });
  const claims = [
    ["Supported", "On these close-ups, a model that sees local structure beats a flat pixel vector. The gap is from about 93% (pixel SVM) to about 99.7% (unfrozen MobileNet) on their own test counts."],
    ["Not this dataset", "The paper's classical models near 50% do not show up here. Crack close-ups are easier than PEER's wide shots, clutter, and bad labels."],
    ["Not a general winner", "InceptionV3 is not the best network in this run. One dataset does not crown an architecture."],
  ];
  claims.forEach((item, i) => {
    const y = 1.2 + i * 1.55;
    addText(s, item[0], {
      x: 0.55, y, w: 3.1, h: 0.45,
      fontSize: 16, bold: true, color: "E7B89A", margin: 0,
    });
    addText(s, item[1], {
      x: 3.7, y, w: 9.0, h: 1.3,
      fontSize: 16, color: WHITE, margin: 0,
    });
  });
  addText(s, "python train.py     then     python demo.py", {
    x: 0.55, y: 6.55, w: 12, h: 0.4,
    fontSize: 16, color: "D7D3CC", margin: 0,
  });
  s.addNotes("End on the claim we can defend. If the panel asks for the paper's 83%, say our number is higher because the pictures are cleaner, and point at the logistic learning curve as evidence that flat pixels plateau.");
}

pres.writeFile({ fileName: "D:/SEM5/ML/LAB/MINI-PROJECT/presentation.pptx" })
  .then(() => console.log("wrote presentation"))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
