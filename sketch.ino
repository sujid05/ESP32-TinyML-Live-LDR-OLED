#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model_data.h"

// Hardware Pin Definitions
#define LDR_PIN 34
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// TensorFlow Lite Micro Objects
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 8 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  const char* LABELS[] = {"DARK", "NORMAL", "BRIGHT"};
}

void setup() {
  Serial.begin(115200);
  pinMode(LDR_PIN, INPUT);

  // Initialize I2C and OLED Display
  Wire.begin(21, 22);
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;);
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("TinyML Initializing...");
  display.display();

  // Load TFLite Model
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  
  delay(1000);
}

void loop() {
  // 1. Read Live Analog Sensor Value
  int rawLdr = analogRead(LDR_PIN);
  float luxEstimate = map(rawLdr, 0, 4095, 0, 2500); // Scale 12-bit ADC to Lux

  // 2. Feed Live Reading into TFLite Model Input Tensor
  input->data.f[0] = luxEstimate;

  // 3. Execute On-Device Inference
  if (interpreter->Invoke() == kTfLiteOk) {
    float dark_prob = output->data.f[0];
    float normal_prob = output->data.f[1];
    float bright_prob = output->data.f[2];

    int predictedClass = 0;
    float maxProb = dark_prob;

    if (normal_prob > maxProb) {
      maxProb = normal_prob;
      predictedClass = 1;
    }
    if (bright_prob > maxProb) {
      maxProb = bright_prob;
      predictedClass = 2;
    }

    // 4. Update OLED Display
    display.clearDisplay();

    // Title Header
    display.setTextSize(1);
    display.setCursor(15, 0);
    display.println("TINYML CLASSIFIER");
    display.drawFastHLine(0, 10, 128, SSD1306_WHITE);

    // Live Sensor Reading
    display.setCursor(0, 16);
    display.print("Light: ");
    display.print((int)luxEstimate);
    display.println(" Lux");

    // TinyML Predicted Class Output
    display.setCursor(0, 30);
    display.print("Class: ");
    display.setTextSize(2);
    display.setCursor(45, 26);
    display.println(LABELS[predictedClass]);

    // Confidence Progress Bar
    display.setTextSize(1);
    display.setCursor(0, 48);
    display.print("Conf: ");
    display.print((int)(maxProb * 100));
    display.println("%");

    int barWidth = map((int)(maxProb * 100), 0, 100, 0, 60);
    display.drawRect(65, 48, 60, 8, SSD1306_WHITE);
    display.fillRect(65, 48, barWidth, 8, SSD1306_WHITE);

    display.display();
  }

  delay(300); // Dynamic sampling rate
}
