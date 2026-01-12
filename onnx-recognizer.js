// See: https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-script-tag

// see also advanced usage of importing ONNX Runtime Web:
// https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web

// import ONNXRuntime Web from CDN
//import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js";
// set wasm path override
//ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// // import ONNXRuntime Web
// import * as ort from "./ort/ort.min.js";
// // set wasm path override -- loads e.g. 'ort-wasm-simd.wasm' -- see: https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/
// ort.env.wasm.wasmPaths = "./ort/";

// Dynamically load ONNX runtime without needing to be in a module context
async function _ensureOnnxLoaded() {
    if (window.ort) {
        return window.ort;
    }

    if (!window.onnxLoadPromise) {
        window.onnxLoadPromise = new Promise((resolve, reject) => {
            console.log("Loading ONNX Runtime...");
            window.onnxLoadResolve = resolve;
            window.onnxLoadReject = reject;
            const script = document.createElement("script");
            script.type = "module";
            // Inject script content
            script.text = `
                //import * as ort from "./ort/ort.min.js";
                try {
                    const ort = await import("./ort/ort.min.js");
                    window.ort = ort;
                    ort.env.wasm.wasmPaths = "./ort/";
                    window.onnxLoadResolve(ort);
                } catch (err) {
                    window.onnxLoadReject(err);
                }
            `;
            // script.onload = () => { resolve(); };
            // script.onerror = (err) => { reject(err); };
            document.head.appendChild(script);
        });
    }

    await window.onnxLoadPromise;
    return window.ort;
}


class OnnxRecognizer {
    constructor(modelFile, inputFormat, inputName, outputName) {
        this.session = null;
        this.modelFile = modelFile;       // './trained_models/whole_model_quickdraw.onnx';
        this.INPUT_FORMAT = inputFormat;  // [1, 1, IMAGE_SIZE, IMAGE_SIZE];  // IMAGE_SIZE = 28;
        this.INPUT_NAME = inputName;      // 'input';
        this.OUTPUT_NAME = outputName;    // 'linear_2';
    }

    async load() {
        await _ensureOnnxLoaded();
        this.session = await ort.InferenceSession.create(this.modelFile);
        return this.session;
    }

    async recognize(input) {
        if (!this.session) {
            throw new Error("ONNX model is not loaded yet, must await a call to load().");
        }
        try {
            // prepare inputs. a tensor need its corresponding TypedArray as data
            // Input 0: input, shape: [1, 1, 28, 28]
            // Output 0: linear_2, shape: [1, 20]
            const dataInput = Float32Array.from(input);
            const tensorInput = new ort.Tensor('float32', dataInput, this.INPUT_FORMAT);

            // prepare feeds. use model input names as keys.
            const feeds = {};
            feeds[this.INPUT_NAME] = tensorInput;

            // feed inputs and run
            const results = await this.session.run(feeds);

            // read from results (Float32Array)
            const rawOutput = results[this.OUTPUT_NAME].data;
            const output = Array.from(rawOutput);
            //console.log(`data of result tensor: ${output}`);

            return output;
        } catch (e) {
            console.error(`failed to inference ONNX model: ${e}.`);
            return null;
        }

    }

    softmax(values) {
        const maxLogit = Math.max(...values);
        const exps = values.map((x) => Math.exp(x - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const softmax = exps.map((v) => v / sumExps);
        //console.log(`softmax: ${softmax}`);
        return softmax;
    }

    orderedIndices(values) {
        const indices = values.map((value, index) => index);
        indices.sort((a, b) => values[b] - values[a]);
        return indices;
    }

    maxIndex(values) {
        let maxIndex = null, maxValue = null;
        for (let i = 0; i < values.length; i++) {
            if (maxIndex === null || values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}

// Hack to export only if imported as a module (top-level await a regexp divided, otherwise an undefined variable divided followed by a comment)
if(0)typeof await/0//0; export default OnnxRecognizer;