let preprocess, detect, detectVideo;
let loading = { loading: true, progress: 0 };
let yolo_model = {};

yolo_model.load = () =>
  tf.ready().then(async () => {
    const yolov8 = await tf.loadGraphModel(`./yolov8n_web_model/model.json`, {
      onProgress: (fractions) => {
        loading = { loading: true, progress: fractions }; // set loading fractions
      },
    }); // load model

    // warming up model
    const dummyInput = tf.ones(yolov8.inputs[0].shape);
    const warmupResults = yolov8.execute(dummyInput);

    loading = { loading: false, progress: 1 };
    yolo_model.model = {
      net: yolov8,
      inputShape: yolov8.inputs[0].shape,
    }; // set model & input shape

    tf.dispose([warmupResults, dummyInput]); // cleanup memory

    return fetch("./labels_waste.json") //fetch("./labels.json")
      .then((response) => response.json())
      .then((labels) => {
        const numClass = labels.length;

        yolo_model.preprocess = (source, modelWidth, modelHeight) => {
          let xRatio, yRatio; // ratios for boxes

          const input = tf.tidy(() => {
            const img = tf.browser.fromPixels(source);

            // padding image to square => [n, m] to [n, n], n > m
            const [h, w] = img.shape.slice(0, 2); // get source width and height
            const maxSize = Math.max(w, h); // get max size
            const imgPadded = img.pad([
              [0, maxSize - h], // padding y [bottom only]
              [0, maxSize - w], // padding x [right only]
              [0, 0],
            ]);

            xRatio = maxSize / w; // update xRatio
            yRatio = maxSize / h; // update yRatio

            return tf.image
              .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
              .div(255.0) // normalize
              .expandDims(0); // add batch
          });

          return [input, xRatio, yRatio];
        };

        /**
         * Function run inference and do detection from source.
         * @param {HTMLImageElement|HTMLVideoElement} source
         * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
         * @param {HTMLCanvasElement} canvasRef canvas reference
         * @param {VoidFunction} callback function to run after detection process
         */
        yolo_model.detect = async (source, model, callback = () => {}) => {
          const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height
          tf.engine().startScope(); // start scoping tf engine
          const [input, xRatio, yRatio] = yolo_model.preprocess(
            source,
            modelWidth,
            modelHeight
          ); // preprocess image

          const res = model.net.execute(input); // inference model
          const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
          const boxes = tf.tidy(() => {
            const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
            const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
            const x1 = tf.sub(
              transRes.slice([0, 0, 0], [-1, -1, 1]),
              tf.div(w, 2)
            ); // x1
            const y1 = tf.sub(
              transRes.slice([0, 0, 1], [-1, -1, 1]),
              tf.div(h, 2)
            ); // y1
            return tf
              .concat(
                [
                  y1,
                  x1,
                  tf.add(y1, h), //y2
                  tf.add(x1, w), //x2
                ],
                2
              )
              .squeeze();
          }); // process boxes [y1, x1, y2, x2]

          const [scores, classes] = tf.tidy(() => {
            // class scores
            const rawScores = transRes
              .slice([0, 0, 4], [-1, -1, numClass])
              .squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
            return [rawScores.max(1), rawScores.argMax(1)];
          }); // get max scores and classes index

          const nms = await tf.image.nonMaxSuppressionAsync(
            boxes,
            scores,
            500,
            0.45,
            0.2
          ); // NMS to filter boxes

          const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
          const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
          const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

          const prediction = [];
          for (let i = 0; i < classes_data.length; i++) {
            let [y1, x1, y2, x2] = boxes_data.slice(i * 4, (i + 1) * 4);
            // y1 *= yRatio;
            // x1 *= xRatio;
            // y2 *= yRatio;
            // x2 *= xRatio;
            const width = x2 - x1;
            const height = y2 - y1;
            prediction.push({
              class: labels[classes_data[i]],
              bbox: [x1, y1, width, height],
              score: scores_data[i],
            });
          }
          tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory

          callback();

          tf.engine().endScope(); // end of scoping
          return prediction;
        };

        /**
         * Function to detect video from every source.
         * @param {HTMLVideoElement} vidSource video source
         * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
         * @param {HTMLCanvasElement} canvasRef canvas reference
         */
        yolo_model.detectVideo = (vidSource, callback) => {
          const model = yolo_model.model;
          /**
           * Function to detect every frame from video
           */

          return yolo_model.detect(vidSource, model);
        };
        return yolo_model;
      });
  });
