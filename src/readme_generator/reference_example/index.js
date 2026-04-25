import React from 'react';
import ConfigGenerator from '../../base/ConfigGenerator';

/**
 * Llama 3.1 Configuration Generator
 */
const Llama31ConfigGenerator = () => {
  const config = {
    modelFamily: 'meta-llama',

    options: {
      hardware: {
        name: 'hardware',
        title: 'Hardware Platform',
        items: [
          { id: 'cpu', label: 'Xeon CPU', default: true }
        ]
      },
      modelsize: {
        name: 'modelsize',
        title: 'Model Size',
        items: [
          { id: '8b', label: '8B', default: true }
        ]
      },
      quantization: {
        name: 'quantization',
        title: 'Quantization',
        items: [
          { id: 'bf16', label: 'BF16', default: true },
          { id: 'w8a8_int8', label: 'W8A8_INT8', default: false },
		  { id: 'fp8', label: 'FP8', default: false },
		  { id: 'awq_int4', label: 'AWQ_INT4', default: false },
        ]
      },
    },

    generateCommand: function(values) {
      const { hardware, modelsize, quantization } = values;

      // Determine model path
      let modelPath;
      if (quantization === 'w8a8_int8') {
		  modelPath = `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8`;
	  } else if (quantization === 'fp8') {
		  modelPath = `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`;
	  } else if (quantization === 'awq_int4') {
		  modelPath = `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`;
	  } else {
		  modelPath = `meta-llama/Llama-3.1-8B-Instruct`;
	  }

      // Build command args
      const args = [];
      args.push(`--model-path ${modelPath}`);
	  args.push(`--trust-remote-code`);
	  args.push(`--disable-overlap-schedule`);
	  args.push(`--device cpu`);
	  if (quantization === 'w8a8_int8') {
		  args.push(`--quantization w8a8_int8`);
	  }
	  args.push(`--enable-torch-compile`);
	  args.push(`--host 0.0.0.0`);
      args.push(`--tp 6`);

      let cmd = 'python -m sglang.launch_server \\\n';
      cmd += `  ${args.join(' \\\n  ')}`;

      return cmd;
    }
  };

  return <ConfigGenerator config={config} />;
};

export default Llama31ConfigGenerator;
