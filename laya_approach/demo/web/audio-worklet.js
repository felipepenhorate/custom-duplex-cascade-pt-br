/**
 * 24 kHz float32 PCM capture worklet.
 *
 * The browser may ignore the requested 24 kHz context rate (Safari often
 * does), so this processor resamples linearly from the actual context rate to
 * 24 kHz and posts 20 ms frames.  The main thread forwards each ArrayBuffer
 * unchanged over the websocket.
 */
class OndaPcm24k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 24000;
    this.ratio = sampleRate / this.targetRate;
    this.queue = [];
    this.readPos = 0;
    this.output = [];
    this.frameSize = 480;
    this.maxQueue = Math.max(this.targetRate, sampleRate) * 3;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const channel = input[0];
      for (let index = 0; index < channel.length; index += 1) {
        this.queue.push(channel[index]);
      }
    }

    while (this.readPos + 1 < this.queue.length) {
      const index = Math.floor(this.readPos);
      const fraction = this.readPos - index;
      const sample = this.queue[index] * (1 - fraction) + this.queue[index + 1] * fraction;
      this.output.push(sample);
      this.readPos += this.ratio;
      if (this.output.length >= this.frameSize) {
        const frame = new Float32Array(this.output.splice(0, this.frameSize));
        this.port.postMessage(frame.buffer, [frame.buffer]);
      }
    }

    const consumed = Math.floor(this.readPos);
    if (consumed > 0) {
      this.queue.splice(0, consumed);
      this.readPos -= consumed;
    }
    if (this.queue.length > this.maxQueue) {
      this.queue.splice(0, this.queue.length - this.targetRate);
      this.readPos = 0;
    }
    return true;
  }
}

registerProcessor('onda-pcm24k', OndaPcm24k);
