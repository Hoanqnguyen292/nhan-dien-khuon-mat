/* ===========================================================================
 *  ESP32-2432S028R (CYD) - MAN HINH CAN MAT + KET QUA DIEM DANH
 *  Firmware: 2026-10-29-cyd6 (giao dien da sap xep lai)
 * ===========================================================================
 *
 *  Vai tro:
 *    1. CHI hien thi anh LIVE trong 15 giay sau khi bam nut tren ESP32-CAM
 *       (MJPEG http://<cam>:81/stream). Khi NGHI KHONG mo stream ->
 *       ESP32-CAM khong chup lien tuc -> giam nhiet, tiet kiem dien.
 *    2. Doc ket qua nhan dien tu ESP32-CAM (GET http://<cam>:80/result)
 *       va hien thi ten / trang thai diem danh (banner).
 *    3. ESP32-CAM bao bat dau/ket thuc qua UDP FACESYNC|SCAN|1|<ip> /
 *       FACESYNC|SCAN|0|<ip> (cong 4321) -> CYD mo/dong stream NGAY,
 *       khong cho 0.5s poll. Neu UDP bi chan, cot `scanning` trong /result
 *       lam duong du phong.
 *
 *  Trang thai man hinh:
 *    - STANDBY (nghi): trang "SAN SANG DIEM DANH", khong truyen anh.
 *    - SCANNING (dang quet): video tu camera + thanh tien trinh + banner ket qua.
 *
 *  Phan cung (xem tft_setup.h):
 *    - ILI9341 240x320 (portrait) SPI HSPI: MOSI=13, MISO=12, SCLK=14,
 *      CS=15, DC=2, RST=-1, BL=21. Man hinh NGANG 320x240 = setRotation(1).
 *    - XPT2046 (VSPI): CS=33, IRQ=36, CLK=25, MOSI=32, MISO=39
 *
 *  Thu vien can cai:
 *    - TFT_eSPI (Bodmer) 2.5.43+
 *    - XPT2046_Touchscreen (Paul Stoffregen) 1.4+
 *    - JPEGDEC (Larry Bank) 1.8.5+
 *
 *  Lenh Serial:
 *    /help, /ip, /ip <ip>, /find, /live on|off, /test, /rot 0..7,
 *    /touch, /stats, /reboot
 *
 * =========================================================================== */

#include <Preferences.h>
#include <SPI.h>
#include <FS.h>                // JPEGDEC dung kieu File -> phai include truoc
#include "tft_setup.h"         // Nap cau hinh ILI9341/CYD truoc khi include TFT_eSPI
#include <TFT_eSPI.h>
#include <XPT2046_Touchscreen.h>
#include <JPEGDEC.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include <esp_heap_caps.h>

// ============================== CAU HINH ====================================
#define FW_VERSION        "2026-11-01-cyd13"

#define CYD_SELFTEST      1    // 1 = tu chay self-test luc boot (~2.5s)

#define WIFI_SSID         "Test"
#define WIFI_PASS         "88888888"

#define CAM_IP_DEFAULT    "192.168.0.101"
#define CAM_HTTP_PORT     80
#define CAM_STREAM_PORT   81

#define RESULT_POLL_MS    500      // doc /result (du phong neu UDP bi chan)
#define STREAM_RETRY_MS   400      // thu lai stream NHANH khi dang quet
// Vong lap quet cua ESP32-CAM bi chan ~2-5s moi frame (cho server nhan dien
// tra loi), nen thong bao UDP chi den moi ~2-5s -> fail-safe phai > 5s.
#define SCAN_NOTIFY_MS    9000     // khong tin hieu quet 9s -> ket thuc (fail-safe)
#define SCAN_MAX_MS       20000    // khong de "scanning" treo mai
#define RESULT_HOLD_MS    4000     // giu khung cuoi sau khi quet xong

#define DISC_PORT         4321
#define DISC_MAGIC        "FACESYNC"

// Cam ung XPT2046 - CYD
#define TOUCH_CS_PIN      33
#define TOUCH_IRQ_PIN     36
#define TOUCH_SCLK_PIN    25
#define TOUCH_MISO_PIN    39
#define TOUCH_MOSI_PIN    32

#define TOUCH_SWAP_XY     0
#define TOUCH_INVERT_X    0
#define TOUCH_INVERT_Y    0

// Vung gia tri RAW cua XPT2046 tren CYD (hieu chuan bang lenh /touch)
#define TOUCH_X_MIN       1500
#define TOUCH_X_MAX       3000
#define TOUCH_Y_MIN       1400
#define TOUCH_Y_MAX       2900

#define LCD_W             320
#define LCD_H             240

// Mau rieng
#define C_BG              0x0841
#define C_HEADER          0x0010
#define C_OK_BG           0x0A64
#define C_ALREADY_BG      0x0010
#define C_ERR_BG          0x7800
#define C_TEXT_DIM        0xA514
#define C_GUIDE           0xDEDB

// ============================== DOI TUONG ===================================
TFT_eSPI tft = TFT_eSPI();

// Rotation mac dinh = 1 (landscape 320x240). Panel CYD la 240x320 doc, nen
// rotation 1 hoac 3 moi ra NGANG; rotation 0/2 la DOC -> UI bi cat con 2/3.
// Co the doi bang /rot <0..7>.
static uint8_t g_rotation = 1;
static uint8_t g_bgr = 0;   // 0 = RGB (mac dinh), 1 = BGR  - doi bang /rgb
static uint8_t g_inv = 0;   // 0 = INVOFF, 1 = INVON       - doi bang /inv

#if CYD_DISPLAY_ON_VSPI
  SPIClass touchSPI(HSPI);
#else
  SPIClass touchSPI(VSPI);
#endif
XPT2046_Touchscreen touch(TOUCH_CS_PIN, TOUCH_IRQ_PIN);

JPEGDEC jpeg;
WiFiClient streamClient;
WiFiUDP udp;
Preferences prefs;
HTTPClient resultHttp;

// ============================== BIEN TRANG THAI =============================
enum StreamState { SS_OFF, SS_CONNECTING, SS_LIVE, SS_RETRY };
static StreamState ss = SS_OFF;

static char camIp[16] = CAM_IP_DEFAULT;
static bool udpReady = false;
static bool resultHttpOpen = false;

static bool manualPaused = false;   // tam dung xem truc tiep (lenh /live off)
static bool forcedScan = false;     // /scan: ep quet de test, khong tu het sau 4s
static bool scanning = false;       // ESP32-CAM dang quet 15s
static uint32_t scanStartMs = 0;
static uint32_t scanEndMs = 0;
static uint32_t lastScanNotifyMs = 0;  // lan cuoi nhan tin hieu "dang quet" (UDP hoac /result)
static bool standbyShown = false;   // da ve trang standby khi nghi

static bool lastFrameValid = false;
static bool idleDirty = true;
static bool wifiDown = false;

static uint32_t lastResultPoll = 0;
static uint32_t lastRetry = 0;
static uint32_t lastDiscovery = 0;
static uint32_t lastWifiTry = 0;
static uint32_t lastFpsMs = 0;
static uint32_t lastStatsMs = 0;
static uint32_t lastScanOverlayMs = 0;

static uint32_t framesThisSec = 0;
static uint32_t framesTotal = 0;
static uint32_t fps = 0;
static uint32_t lastDecodeMs = 0;
static size_t lastFrameBytes = 0;

// Bo dem JPEG (cap phat 1 lan luc boot)
static uint8_t *jpegBuf = nullptr;
static size_t jpegCap = 0;

// Trang thai doc MJPEG chunked
static bool streamChunked = false;
static size_t chunkLeft = 0;
static bool chunkNeedCRLF = false;

// Ket qua nhan dien
static uint32_t lastSeq = 0;
static bool haveSeq = false;
static char resNameAscii[64] = "";

// ============================== KHAI BAO HAM ================================
static void applyRotation(uint8_t r);
static void drawSelfTest();
static void drawStandbyPage();
static void drawOverlay();
static void drawIdlePage();
static void drawStatusPage(const char *title, const char *sub, uint16_t accent);
static void streamStop();
static bool streamConnect();
static int readFrame(size_t *outLen);
static bool showJpeg(size_t len);
static bool pollResult();
static void applyResult(const char *json);
static void startScan(const char *why);
static void endScan(const char *why);
static void pollUdp();
static bool discoverCam(uint16_t waitMs);
static bool wifiConnect(uint32_t timeoutMs);
static void serialPoll();
static void touchPoll();
static void handleSerialLine(char *line);
static void printStats();
static void drawConnBox(const char *title, const char *sub);
static void drawRotCard(uint8_t r);
static void runOrientationWizard();

// ============================== HIEN THI ====================================
static void drawCenteredFit(const char *text, int16_t x, int16_t y,
                            int16_t maxWidth, uint8_t font,
                            uint16_t fg, uint16_t bg) {
  if (!text) return;
  while (font > 1 && tft.textWidth(text, font) > maxWidth) font--;
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(fg, bg);
  tft.drawString(text, x, y, font);
}

static void drawProgress() {
  uint32_t elapsed = millis() - scanStartMs;
  if (elapsed > 15000) elapsed = 15000;
  const int width = (int)((elapsed * 300UL) / 15000UL);
  tft.fillRoundRect(10, 207, 300, 6, 3, 0x2104);
  if (width > 0) tft.fillRoundRect(10, 207, width, 6, 3, TFT_ORANGE);
}

// (Da bo bang ten/ket qua: CYD chi hien hinh camera de can khung.
//  Ten hoc sinh do man LCD rieng hien thi.)

static void drawTopBar() {
  tft.fillRect(0, 0, LCD_W, 28, C_HEADER);

  uint16_t dot = TFT_YELLOW;
  const char *state = "SAN SANG";
  if (scanning) {
    dot = TFT_ORANGE;
    state = "DANG QUET";
  } else if (ss == SS_LIVE) {
    dot = TFT_GREEN;
    state = "TRUC TIEP";
  }

  tft.fillCircle(10, 14, 4, dot);
  tft.setTextDatum(TL_DATUM);
  tft.setTextColor(TFT_WHITE, C_HEADER);
  tft.drawString(state, 20, 7, 2);

  if (scanning) {
    drawCenteredFit("GIU MAT TRONG KHUNG", 190, 14, 220, 1, TFT_ORANGE, C_HEADER);
  } else {
    drawCenteredFit(camIp, 190, 14, 220, 1, C_TEXT_DIM, C_HEADER);
  }
}

static void drawGuide() {
  const uint16_t color = scanning ? TFT_ORANGE : C_GUIDE;
  const int left = 58, right = 262, top = 40, bottom = 197, len = 22;
  for (int i = 0; i < 2; ++i) {
    tft.drawFastHLine(left, top + i, len, color);
    tft.drawFastVLine(left + i, top, len, color);
    tft.drawFastHLine(right - len, top + i, len, color);
    tft.drawFastVLine(right - i, top, len, color);
    tft.drawFastHLine(left, bottom - i, len, color);
    tft.drawFastVLine(left + i, bottom - len, len, color);
    tft.drawFastHLine(right - len, bottom - i, len, color);
    tft.drawFastVLine(right - i, bottom - len, len, color);
  }
}

static void drawBottomBar() {
  tft.fillRect(0, 216, LCD_W, 24, C_HEADER);
  // Tren man hinh CHI huong dan can khung - ten/ket qua diem danh do man LCD
  // rieng hien thi, khong hien tren CYD nua.
  const char *message = scanning
      ? "DANG CHUP  |  GIU MAT TRONG KHUNG"
      : "Bam nut tren camera de bat dau";
  drawCenteredFit(message, LCD_W / 2, 228, LCD_W - 16,
                  2, C_TEXT_DIM, C_HEADER);
}

static void drawOverlay() {
  drawTopBar();
  drawGuide();
  drawBottomBar();

  if (scanning) drawProgress();
}

static void drawSplash(const char *message) {
  tft.fillScreen(C_BG);
  drawCenteredFit("DIEM DANH KHUON MAT", LCD_W / 2, 78,
                  LCD_W - 24, 4, TFT_CYAN, C_BG);
  drawCenteredFit(message, LCD_W / 2, 126,
                  LCD_W - 24, 2, TFT_LIGHTGREY, C_BG);
  drawCenteredFit(FW_VERSION, LCD_W / 2, 218,
                  LCD_W - 24, 1, TFT_DARKGREY, C_BG);
}

// Ghi lai MADCTL voi bit thu tu mau theo g_bgr. TFT_eSPI luon ghi MADCTL
// theo hang so luc build, nen can ghi de khi nguoi dung doi bang /rgb.
static void writeMadctlColor() {
  uint8_t v;
  switch (g_rotation & 7) {
    case 0:  v = 0x40; break;   // MX
    case 1:  v = 0x20; break;   // MV
    case 2:  v = 0x80; break;   // MY
    case 3:  v = 0xE0; break;   // MX|MY|MV
    case 4:  v = 0xC0; break;   // MX|MY
    case 5:  v = 0x60; break;   // MX|MV
    case 6:  v = 0x00; break;   // (khong lat)
    default: v = 0xA0; break;   // MY|MV
  }
  if (g_bgr) v |= 0x08;         // bit BGR
  tft.writecommand(0x36);       // MADCTL
  tft.writedata(v);
}

static void applyRotation(uint8_t rotation) {
  g_rotation = rotation & 7;
  tft.setRotation(g_rotation);
  writeMadctlColor();           // ap lai thu tu mau theo g_bgr
  Serial.printf("[TFT] rotation %u -> %dx%d\n",
                (unsigned)g_rotation, tft.width(), tft.height());
}

static void drawSelfTest() {
  const uint16_t colors[] = {TFT_RED, TFT_GREEN, TFT_BLUE, TFT_WHITE};
  for (uint8_t i = 0; i < 4; ++i) {
    tft.fillScreen(colors[i]);
    delay(180);
  }

  tft.fillScreen(TFT_BLACK);
  tft.drawRect(0, 0, LCD_W, LCD_H, TFT_WHITE);
  tft.drawRect(1, 1, LCD_W - 2, LCD_H - 2, TFT_WHITE);

  tft.setTextFont(1);
  tft.setTextDatum(TL_DATUM);
  tft.setTextColor(TFT_GREEN, TFT_BLACK);
  tft.drawString("0,0", 4, 4, 1);
  tft.setTextDatum(TR_DATUM);
  tft.drawString("319,0", LCD_W - 4, 4, 1);
  tft.setTextDatum(BL_DATUM);
  tft.drawString("0,239", 4, LCD_H - 4, 1);
  tft.setTextDatum(BR_DATUM);
  tft.drawString("319,239", LCD_W - 4, LCD_H - 4, 1);

  char label[32];
  snprintf(label, sizeof(label), "SELF TEST %dx%d", tft.width(), tft.height());
  drawCenteredFit(label, LCD_W / 2, LCD_H / 2 - 10,
                  LCD_W - 24, 4, TFT_YELLOW, TFT_BLACK);
  char rotationLabel[40];
  snprintf(rotationLabel, sizeof(rotationLabel), "Rotation %u",
           (unsigned)g_rotation);
  drawCenteredFit(rotationLabel, LCD_W / 2, LCD_H / 2 + 24,
                  LCD_W - 24, 1, TFT_LIGHTGREY, TFT_BLACK);
  delay(1200);
}

static void drawHeader() {
  tft.fillRect(0, 0, LCD_W, 32, C_HEADER);
  tft.setTextDatum(TL_DATUM);
  tft.setTextColor(TFT_CYAN, C_HEADER);
  tft.drawString("DIEM DANH", 12, 8, 2);
}

static void drawFooter() {
  tft.fillRect(0, 211, LCD_W, 29, C_HEADER);
  char status[64];
  const int rssi = WiFi.status() == WL_CONNECTED ? (int)WiFi.RSSI() : 0;
  snprintf(status, sizeof(status), "CAM %s   |   WIFI %d dBm", camIp, rssi);
  drawCenteredFit(status, LCD_W / 2, 226,
                  LCD_W - 16, 1, C_TEXT_DIM, C_HEADER);
}

static void drawStandbyPage() {
  tft.fillScreen(C_BG);
  drawHeader();

  const uint16_t icon = TFT_GREEN;
  tft.drawRoundRect(129, 52, 62, 48, 9, icon);
  tft.drawCircle(160, 76, 13, icon);
  tft.drawCircle(160, 76, 6, icon);
  tft.fillRoundRect(139, 46, 20, 8, 3, icon);
  tft.fillCircle(181, 60, 2, icon);

  drawCenteredFit("SAN SANG DIEM DANH", LCD_W / 2, 126,
                  LCD_W - 24, 4, TFT_WHITE, C_BG);
  drawCenteredFit("Bam nut tren ESP32-CAM", LCD_W / 2, 157,
                  LCD_W - 32, 2, C_TEXT_DIM, C_BG);
  drawCenteredFit("de chup anh va nhan dien", LCD_W / 2, 179,
                  LCD_W - 32, 2, C_TEXT_DIM, C_BG);

  drawFooter();
}

static void drawIdlePage() {
  drawStandbyPage();
}

static void drawStatusPage(const char *title, const char *sub, uint16_t accent) {
  tft.fillScreen(C_BG);
  drawHeader();

  tft.fillRoundRect(24, 67, 272, 112, 12, C_HEADER);
  tft.drawRoundRect(24, 67, 272, 112, 12, accent);
  drawCenteredFit(title, LCD_W / 2, 108,
                  244, 4, accent, C_HEADER);
  drawCenteredFit(sub, LCD_W / 2, 145,
                  244, 2, TFT_LIGHTGREY, C_HEADER);

  drawFooter();
}

// ============================== WIZARD HUONG MAN HINH =======================
// The thu: khung trang + mui ten chi huong LEN cua khung UI + so rotation.
static void drawRotCard(uint8_t r) {
  const int w = tft.width();
  const int h = tft.height();

  tft.fillScreen(TFT_BLACK);
  for (int i = 0; i < 3; ++i) {
    tft.drawRect(i, i, w - 2 * i, h - 2 * i, TFT_WHITE);
  }

  // Mui ten chi "huong len" theo khung toa do UI dang dung
  const int cx = w / 2;
  const int ay = h / 2 - 46;
  tft.fillTriangle(cx, ay, cx - 24, ay + 34, cx + 24, ay + 34, TFT_RED);
  tft.fillRect(cx - 9, ay + 34, 18, 46, TFT_RED);

  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_YELLOW, TFT_BLACK);
  char b[32];
  snprintf(b, sizeof(b), "ROT %u", (unsigned)r);
  tft.drawString(b, cx, h / 2 + 22, 4);
  snprintf(b, sizeof(b), "%dx%d", w, h);
  tft.drawString(b, cx, h / 2 + 58, 2);
  tft.setTextColor(TFT_GREEN, TFT_BLACK);
  tft.drawString("Chu doc ngang + vien kin 4 canh = huong nay dung",
                 cx, h - 18, 1);
}

// Xoay lan luot rotation 0..7, moi buoc 2 giay, de nguoi dung tu chon.
static void runOrientationWizard() {
  const uint8_t prev = g_rotation;
  Serial.println("[TFT] WIZARD: thu lan luot rot 0..7, moi buoc 2 giay.");
  Serial.println("[TFT] MEO: chi rot 1, 3, 5, 7 moi NGANG va kin man hinh;");
  Serial.println("[TFT]      0, 2, 4, 6 la DOC -> se thay dai nhieu trang phia tren.");
  Serial.println("[TFT] Khi thay chu doc NGANG, ro va vien trang kin 4 canh");
  Serial.println("[TFT] thi ghi so o dong 'ROT n', roi go: /rot n  (de luu).");

  for (uint8_t r = 0; r < 8; ++r) {
    applyRotation(r);
    drawRotCard(r);
    Serial.printf("[TFT]   thu rot %u -> %dx%d\n", (unsigned)r,
                  tft.width(), tft.height());
    delay(2000);
  }

  applyRotation(prev);
  drawStatusPage("XOAY MAN HINH", "Da thu xong 0..7 - go /rot N", TFT_CYAN);
  Serial.printf("[TFT] Wizard xong. Quay lai rot %u.\n", (unsigned)prev);
}

// ============================== JPEG DECODE =================================
static int16_t drawOffX = 0;
static int16_t drawOffY = 0;
static int16_t imgOutW = 0;     // kich thuoc anh SAU scale (truoc khi ve)
static int16_t imgOutH = 0;

// Xoay/lat HINH CAMERA chi tren CYD. KHONG doi huong sensor ben ESP32-CAM,
// nen anh gui len server nhan dien van dung huong.
//   0 = giu nguyen   1 = xoay 180 do (sua loi hinh bi nguoc)
//   2 = lat ngang    3 = lat doc
static uint8_t g_imgRot = 1;

static const char *imgRotName(uint8_t r) {
  switch (r) {
    case 1:  return "xoay 180";
    case 2:  return "lat ngang";
    case 3:  return "lat doc";
    default: return "giu nguyen";
  }
}

// Ve mot khoi pixel (da cat theo kich thuoc man hinh)
static void drawBlock(int32_t x, int32_t y, int32_t stride,
                      int32_t w, int32_t h, const uint16_t *px) {
  x += drawOffX;
  y += drawOffY;
  if (x >= LCD_W || y >= LCD_H || x + w <= 0 || y + h <= 0) return;

  int32_t drawW = w;
  int32_t drawH = h;
  if (x + drawW > LCD_W) drawW = LCD_W - x;
  if (y + drawH > LCD_H) drawH = LCD_H - y;
  if (drawW <= 0 || drawH <= 0) return;

  if (drawW == stride) {
    tft.pushImage(x, y, drawW, drawH, px);
  } else {
    for (int32_t r = 0; r < drawH; ++r) {
      tft.pushImage(x, y + r, drawW, 1, px + (size_t)r * stride);
    }
  }
}

// Bo dem tam cho 1 khoi MCU (toi da 16x16 = 256 pixel)
static uint16_t mcuTmp[16 * 16];

static int drawMCUs(JPEGDRAW *pDraw) {
  const int32_t stride = pDraw->iWidth;
  int32_t w = pDraw->iWidthUsed ? pDraw->iWidthUsed : stride;
  const int32_t h = pDraw->iHeight;
  if (stride <= 0 || w <= 0 || h <= 0) return 1;
  if (w > stride) w = stride;

  // Khong xoay/lat, hoac khoi to hon du kien: ve thang
  if (g_imgRot == 0 || w > 16 || h > 16 || imgOutW <= 0 || imgOutH <= 0) {
    static uint8_t dbgFlat = 0;
    if (dbgFlat < 3) {
      ++dbgFlat;
      Serial.printf("[IMG] block x=%d y=%d stride=%d w=%d h=%d rot=%u -> VE THANG (out=%dx%d)\n",
                    (int)pDraw->x, (int)pDraw->y, (int)stride, (int)w, (int)h,
                    (unsigned)g_imgRot, (int)imgOutW, (int)imgOutH);
    }
    drawBlock(pDraw->x, pDraw->y, stride, w, h, pDraw->pPixels);
    return 1;
  }

  static uint8_t dbgRot = 0;
  if (dbgRot < 3) {
    ++dbgRot;
    Serial.printf("[IMG] block x=%d y=%d stride=%d w=%d h=%d rot=%u -> XOAY/LAT (out=%dx%d)\n",
                  (int)pDraw->x, (int)pDraw->y, (int)stride, (int)w, (int)h,
                  (unsigned)g_imgRot, (int)imgOutW, (int)imgOutH);
  }

  // Vi tri khoi sau khi xoay/lat (trong khung anh goc)
  int32_t tx = pDraw->x;
  int32_t ty = pDraw->y;
  if (g_imgRot == 1)      { tx = imgOutW - pDraw->x - w; ty = imgOutH - pDraw->y - h; }
  else if (g_imgRot == 2) { tx = imgOutW - pDraw->x - w; }
  else if (g_imgRot == 3) { ty = imgOutH - pDraw->y - h; }
  if (tx < 0 || ty < 0) return 1;

  const uint16_t *src = pDraw->pPixels;
  for (int32_t r = 0; r < h; ++r) {
    const int32_t sr = (g_imgRot == 1 || g_imgRot == 3) ? (h - 1 - r) : r;
    uint16_t *dst = mcuTmp + (size_t)r * 16;
    for (int32_t c = 0; c < w; ++c) {
      const int32_t sc = (g_imgRot == 1 || g_imgRot == 2) ? (w - 1 - c) : c;
      dst[c] = src[(size_t)sr * stride + sc];
    }
  }

  // Khoi rong dung 16 pixel -> du lieu lien tuc trong mcuTmp -> ve 1 lan
  if (w == 16) {
    tft.pushImage(tx + drawOffX, ty + drawOffY, 16, h, mcuTmp);
  } else {
    for (int32_t r = 0; r < h; ++r) {
      tft.pushImage(tx + drawOffX, ty + r + drawOffY, w, 1,
                    mcuTmp + (size_t)r * 16);
    }
  }
  return 1;
}

static bool showJpeg(size_t len) {
  if (!jpeg.openRAM(jpegBuf, (int)len, drawMCUs)) {
    Serial.println("[JPEG] openRAM that bai");
    return false;
  }

  int w = jpeg.getWidth();
  int h = jpeg.getHeight();
  int opt = 0;
  int shift = 0;
  if (w >= 1120) { opt = JPEG_SCALE_QUARTER; shift = 2; }
  else if (w >= 560) { opt = JPEG_SCALE_HALF; shift = 1; }

  int outW = w >> shift;
  int outH = h >> shift;
  imgOutW = (int16_t)outW;
  imgOutH = (int16_t)outH;
  drawOffX = (outW < LCD_W) ? (int16_t)((LCD_W - outW) / 2) : 0;
  drawOffY = (outH < LCD_H) ? (int16_t)((LCD_H - outH) / 2) : 0;

  Serial.printf("[JPEG] %dx%d -> scale=%d out=%dx%d off=%d,%d\n",
                w, h, opt, outW, outH, drawOffX, drawOffY);

  if (outW < LCD_W || outH < LCD_H) {
    tft.startWrite();
    tft.fillRect(drawOffX, drawOffY, outW, outH, TFT_BLACK);
    tft.endWrite();
  }

  jpeg.setPixelType(RGB565_LITTLE_ENDIAN);
  uint32_t t0 = micros();
  int ok = jpeg.decode(0, 0, opt);
  lastDecodeMs = (micros() - t0) / 1000;
  jpeg.close();

  if (!ok) {
    Serial.println("[JPEG] decode that bai");
    return false;
  }
  return true;
}

// ============================== STREAM MJPEG ================================
static void streamStop() {
  if (streamClient.connected()) streamClient.stop();
  streamClient.stop();
  ss = SS_OFF;
  streamChunked = false;
  chunkLeft = 0;
  chunkNeedCRLF = false;
}

static int streamReadByte(uint32_t timeoutMs) {
  uint32_t t0 = millis();
  while (true) {
    int n = streamClient.available();
    if (n > 0) return streamClient.read();
    if (!streamClient.connected()) return -1;
    if (millis() - t0 > timeoutMs) return -2;
    delay(1);
  }
}

static int streamBodyByte(uint32_t timeoutMs) {
  if (!streamChunked) return streamReadByte(timeoutMs);

  for (;;) {
    if (chunkLeft == 0) {
      if (chunkNeedCRLF) {
        int a = streamReadByte(timeoutMs);
        int b = streamReadByte(timeoutMs);
        if (a < 0 || b < 0) return -1;
        chunkNeedCRLF = false;
      }
      char line[24];
      int n = 0;
      for (;;) {
        int c = streamReadByte(timeoutMs);
        if (c < 0) return -1;
        if (c == '\n') break;
        if (c != '\r' && n < (int)sizeof(line) - 1) line[n++] = (char)c;
      }
      line[n] = '\0';
      if (n == 0) continue;
      char *end = nullptr;
      unsigned long sz = strtoul(line, &end, 16);
      if (end == line) return -1;
      if (sz == 0) return -1;
      chunkLeft = (size_t)sz;
    }
    int b = streamReadByte(timeoutMs);
    if (b < 0) return -1;
    if (--chunkLeft == 0) chunkNeedCRLF = true;
    return b;
  }
}

static bool streamConnect() {
  streamStop();
  if (WiFi.status() != WL_CONNECTED) return false;

  Serial.printf("[%lums] [CAM] Ket noi stream %s:%d ...\n",
                (unsigned long)millis(), camIp, CAM_STREAM_PORT);
  IPAddress addr;
  if (!addr.fromString(camIp)) {
    Serial.printf("[CAM] IP khong hop le: %s\n", camIp);
    return false;
  }
  if (!streamClient.connect(addr, CAM_STREAM_PORT, 1500)) {
    Serial.println("[CAM] TCP connect that bai");
    return false;
  }
  streamClient.setNoDelay(true);
  streamClient.setTimeout(2000);

  streamClient.print("GET /stream HTTP/1.1\r\nHost: ");
  streamClient.print(camIp);
  streamClient.print(":");
  streamClient.print(CAM_STREAM_PORT);
  streamClient.print("\r\nUser-Agent: CYD-FaceSync\r\nAccept: */*\r\n"
                     "Connection: close\r\n\r\n");

  char hdr[1024];
  size_t hn = 0;
  bool done = false;
  uint32_t t0 = millis();
  while (millis() - t0 < 5000) {
    while (streamClient.available()) {
      char c = (char)streamClient.read();
      if (hn < sizeof(hdr) - 1) hdr[hn++] = c;
      if (hn >= 4 &&
          hdr[hn - 4] == '\r' && hdr[hn - 3] == '\n' &&
          hdr[hn - 2] == '\r' && hdr[hn - 1] == '\n') {
        done = true;
        break;
      }
    }
    if (done) break;
    if (!streamClient.connected()) break;
    delay(1);
  }
  hdr[hn] = '\0';

  if (!done || !strstr(hdr, " 200 ")) {
    Serial.printf("[CAM] Header loi (done=%d): %.80s\n", (int)done, hdr);
    streamClient.stop();
    return false;
  }

  streamChunked = (strstr(hdr, "chunked") != NULL) ||
                  (strstr(hdr, "Chunked") != NULL);
  chunkLeft = 0;
  chunkNeedCRLF = false;
  Serial.printf("[CAM] Stream OK (%s)\n", streamChunked ? "chunked" : "raw");
  return true;
}

static int readFrame(size_t *outLen) {
  size_t n = 0;
  bool in = false;
  bool overflow = false;
  int prev = -1;
  uint32_t t0 = millis();

  // Trong dot quet, CAM chi gui frame moi ~2-5s (vong lap quet bi chan boi
  // POST nhan dien) -> cho toi 7s moi coi la stream chet.
  while (millis() - t0 < 12000) {
    int b = streamBodyByte(7000);
    if (b < 0) return -1;

    if (!in) {
      if (prev == 0xFF && b == 0xD8) {          // SOI
        in = true;
        n = 0;
        overflow = false;
        if (jpegCap >= 2) {
          jpegBuf[n++] = 0xFF;
          jpegBuf[n++] = 0xD8;
        } else {
          overflow = true;
        }
      }
      prev = b;
      continue;
    }

    if (prev == 0xFF && b == 0xD8) {          // SOI moi truoc EOI -> bo cu, bat dau lai
      n = 0;
      overflow = false;
      if (jpegCap >= 2) {
        jpegBuf[n++] = 0xFF;
        jpegBuf[n++] = 0xD8;
      } else {
        overflow = true;
      }
      prev = b;
      continue;
    }

    if (n < jpegCap) jpegBuf[n++] = (uint8_t)b;
    else overflow = true;

    if (prev == 0xFF && b == 0xD9) {          // EOI
      if (overflow) {
        Serial.printf("[%lums] [CAM] Bo frame > %u bytes (thieu bo dem)\n",
                      (unsigned long)millis(), (unsigned)jpegCap);
        return 0;
      }
      *outLen = n;
      return 1;
    }
    prev = b;
  }
  return -1;
}

// ============================== HTTP / RESULT ===============================
static void resultHttpClose() {
  if (resultHttpOpen) {
    resultHttp.end();
    resultHttpOpen = false;
  }
}

static bool pollResult() {
  if (WiFi.status() != WL_CONNECTED) return false;

  // Timeout NGAN + khong dung lai ket noi (setReuse false) de 1 lan poll loi
  // khong chan vong lap chinh (truoc day connect 700ms + doc 1100ms lam nghen
  // loop -> stream :81 khong kip mo trong cua so quet).
  if (!resultHttpOpen) {
    resultHttp.setConnectTimeout(400);
    resultHttp.setTimeout(900);
    resultHttp.setReuse(false);
    if (!resultHttp.begin(String(camIp), CAM_HTTP_PORT, "/result")) {
      return false;
    }
    resultHttpOpen = true;
  }

  const int code = resultHttp.GET();
  if (code == 200) {
    String body = resultHttp.getString();
    if (body.length() > 0 && body.length() < 2048) {
      applyResult(body.c_str());
      return true;
    }
    Serial.printf("[CAM] /result tra ve rong (code=%d, len=%u)\n",
                  code, (unsigned)body.length());
  } else if (code <= 0) {
    resultHttpClose();
  } else {
    Serial.printf("[CAM] /result HTTP code=%d\n", code);
    resultHttpClose();
  }
  return false;
}

// Helpers JSON ngan (tranh dang String de tiet kiem heap)
static int jsonIntField(const char *json, const char *key, int defVal) {
  char keybuf[32];
  snprintf(keybuf, sizeof(keybuf), "\"%s\":", key);
  const char *p = strstr(json, keybuf);
  if (!p) return defVal;
  p += strlen(keybuf);
  while (*p == ' ') p++;
  int sign = 1, val = 0;
  if (*p == '-') { sign = -1; p++; }
  while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
  return sign * val;
}

static void jsonStrField(const char *json, const char *key, char *out, size_t outSz) {
  out[0] = '\0';
  if (outSz == 0) return;
  char keybuf[32];
  snprintf(keybuf, sizeof(keybuf), "\"%s\":\"", key);
  const char *p = strstr(json, keybuf);
  if (!p) return;
  p += strlen(keybuf);
  size_t i = 0;
  while (*p && *p != '"' && i < outSz - 1) {
    out[i++] = *p++;
  }
  out[i] = '\0';
}

static void startScan(const char *why) {
  if (scanning) return;
  scanning = true;
  scanStartMs = millis();
  scanEndMs = 0;
  lastScanNotifyMs = scanStartMs;
  standbyShown = false;
  lastFrameValid = false;
  lastScanOverlayMs = 0;
  lastRetry = 0;
  Serial.printf("[%lums] [CAM] Bat dau quet (%s) -> MO stream :81\n",
                (unsigned long)millis(), why);
}

static void endScan(const char *why) {
  if (!scanning) return;
  scanning = false;
  forcedScan = false;
  scanEndMs = millis();
  Serial.printf("[%lums] [CAM] Ket thuc quet (%s) -> DONG stream de ha nhiet\n",
                (unsigned long)millis(), why);
  streamStop();
  ss = SS_OFF;
}

static void applyResult(const char *json) {
  bool scan = jsonIntField(json, "scanning", 0) != 0;
  if (scan) {
    lastScanNotifyMs = millis();
    if (!scanning) startScan("result");
  } else if (scanning && !forcedScan) {
    // Bo qua ket qua /result cu (gui truoc khi bam nut). Chi ket thuc neu da
    // lau khong con tin hieu quet nao (UDP SCAN|0 se ket thuc ngay lap tuc).
    // /scan (forcedScan) KHONG bi /result ket thuc -> du de test video.
    if (lastScanNotifyMs == 0 ||
        (int32_t)(millis() - lastScanNotifyMs) > 1200) {
      endScan("result");
    }
  }

  int seq = jsonIntField(json, "seq", -1);
  if (seq >= 0) {
    if (haveSeq && (uint32_t)seq == lastSeq) return;
    bool first = !haveSeq;
    lastSeq = (uint32_t)seq;
    haveSeq = true;
    if (first) return;
  }

  char status[16] = "";
  jsonStrField(json, "status", status, sizeof(status));
  bool already = jsonIntField(json, "already", 0) != 0;

  // LUU Y: CYD KHONG hien ten/ket qua diem danh nua (man LCD rieng lo viec do).
  // CYD chi hien hinh camera de hoc sinh can khung. Ket qua chi ghi ra Serial.
  if (!strcmp(status, "ok") || !strcmp(status, "already") || already) {
    char raw[64] = "", cls[24] = "";
    jsonStrField(json, "full_name", raw, sizeof(raw));
    if (!raw[0]) jsonStrField(json, "name", raw, sizeof(raw));
    jsonStrField(json, "class", cls, sizeof(cls));
    Serial.printf("[RES] %s | %s | %s\n", raw[0] ? raw : "?",
                  already || !strcmp(status, "already") ? "DA DIEM DANH"
                                                        : "DIEM DANH OK",
                  cls[0] ? cls : "-");
  } else if (!strcmp(status, "nomatch")) {
    Serial.println("[RES] Khong khop");
  } else if (!strcmp(status, "camerr")) {
    Serial.println("[RES] Loi camera");
  } else if (!strcmp(status, "neterr")) {
    Serial.println("[RES] Loi mang/server");
    Serial.println("[RES] Loi mang/server");
  }
}

// ============================== UDP DISCOVERY ===============================
static bool isIp4Lan(const char *ip) {
  if (!ip || !*ip) return false;
  int o[4] = {0, 0, 0, 0};
  int n = 0, cur = 0, digits = 0;
  for (const char *q = ip; *q && n < 4; q++) {
    char c = *q;
    if (c >= '0' && c <= '9') {
      cur = cur * 10 + (c - '0');
      digits++;
      if (cur > 255 || digits > 3) return false;
    } else if (c == '.') {
      if (digits == 0) return false;
      o[n++] = cur; cur = 0; digits = 0;
    } else break;
  }
  if (digits == 0) return false;
  o[n++] = cur;
  if (n != 4) return false;
  if (o[0] == 127 || o[0] == 0 || o[0] >= 224) return false;
  return (o[0] == 10) || (o[0] == 192 && o[1] == 168) ||
         (o[0] == 172 && o[1] >= 16 && o[1] <= 31);
}

static void pollUdp() {
  if (!udpReady || WiFi.status() != WL_CONNECTED) return;

  for (int i = 0; i < 8; i++) {
    int n = udp.parsePacket();
    if (n <= 0) break;
    char buf[128];
    int len = udp.read(buf, sizeof(buf) - 1);
    if (len <= 0) continue;
    buf[len] = '\0';

    const char *p = strstr(buf, DISC_MAGIC "|SCAN|");
    if (!p) continue;
    p += strlen(DISC_MAGIC) + 6;          // bo "FACESYNC|SCAN|"
    bool on = (*p == '1');

    IPAddress from = udp.remoteIP();
    char fip[16];
    snprintf(fip, sizeof(fip), "%u.%u.%u.%u", from[0], from[1], from[2], from[3]);
    if (!isIp4Lan(fip)) continue;

    if (strcmp(fip, camIp) != 0) {
      strncpy(camIp, fip, sizeof(camIp) - 1);
      camIp[sizeof(camIp) - 1] = '\0';
      prefs.putString("camip", camIp);
      resultHttpClose();
      streamStop();
      Serial.printf("[DISC] SCAN tu IP moi -> cap nhat camera: %s\n", camIp);
    }

    lastScanNotifyMs = millis();
    if (on) {
      if (!scanning) startScan("udp");
    } else if (scanning) {
      endScan("udp");
    }
  }
}

static bool discoverCam(uint16_t waitMs) {
  if (!udpReady || WiFi.status() != WL_CONNECTED) return false;

  udp.beginPacket(IPAddress(255, 255, 255, 255), DISC_PORT);
  udp.print(DISC_MAGIC "|CAM?");
  udp.endPacket();

  uint32_t t0 = millis();
  while (millis() - t0 < waitMs) {
    int n = udp.parsePacket();
    if (n > 0) {
      char buf[128];
      int len = udp.read(buf, sizeof(buf) - 1);
      if (len <= 0) continue;
      buf[len] = '\0';
      const char *p = strstr(buf, DISC_MAGIC "|CAM|");
      if (!p) continue;
      p += strlen(DISC_MAGIC) + 5;
      char ip[16];
      size_t k = 0;
      while (*p && *p != '|' && *p != '\r' && *p != '\n' && k < sizeof(ip) - 1) {
        ip[k++] = *p++;
      }
      ip[k] = '\0';
      if (!isIp4Lan(ip)) continue;
      if (strcmp(ip, camIp) != 0) {
        strncpy(camIp, ip, sizeof(camIp) - 1);
        camIp[sizeof(camIp) - 1] = '\0';
        prefs.putString("camip", camIp);
        resultHttpClose();
        Serial.printf("[DISC] Tim thay ESP32-CAM: %s\n", camIp);
      }
      return true;
    }
    delay(10);
  }
  Serial.println("[DISC] Khong thay ESP32-CAM tra loi");
  return false;
}

// ============================== WIFI ========================================
static bool wifiConnect(uint32_t timeoutMs) {
  if (WiFi.status() == WL_CONNECTED) return true;

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.setHostname("cyd-face");
  Serial.printf("[WiFi] Ket noi '%s' ...", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < timeoutMs) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] That bai");
    return false;
  }
  Serial.printf("[WiFi] OK - IP %s, RSSI %d dBm\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
  return true;
}

// ============================== SERIAL / LENH ===============================
static char rxLine[48];
static size_t rxLen = 0;

static void printStats() {
  Serial.printf("[STATS] fw=%s ip=%s wifi=%d dBm heap=%u\n",
                FW_VERSION, camIp, (int)WiFi.RSSI(), ESP.getFreeHeap());
  Serial.printf("[STATS] stream=%d fps=%u frame=%u B decode=%u ms\n",
                (int)ss, (unsigned)fps, (unsigned)lastFrameBytes,
                (unsigned)lastDecodeMs);
  Serial.printf("[STATS] scanning=%d paused=%d jpegCap=%u\n",
                (int)scanning, (int)manualPaused, (unsigned)jpegCap);
  Serial.printf("[STATS] imgrot=%u (%s) rot=%u %dx%d\n",
                (unsigned)g_imgRot, imgRotName(g_imgRot),
                (unsigned)g_rotation, tft.width(), tft.height());
}

static void handleSerialLine(char *line) {
  while (*line == ' ') line++;
  if (!*line) return;

  if (!strcmp(line, "/help") || !strcmp(line, "help")) {
    Serial.println("Lenh:");
    Serial.println("  /ip                 xem IP camera");
    Serial.println("  /ip <ip>            dat IP camera (luu NVS)");
    Serial.println("  /find               tim camera qua UDP 4321");
    Serial.println("  /live on|off        bat/tat xem truc tiep");
    Serial.println("  /scan               ep mo stream 20s de test video (khong can bam nut)");
    Serial.println("  /test               chay lai self-test man hinh");
    Serial.println("  /rot 0..7           doi rotation va LUU lai (1 hoac 3 = ngang 320x240)");
    Serial.println("  /rot                xem rotation dang dung");
    Serial.println("  /wizard             tu thu rot 0..7 de tim huong man hinh dung");
    Serial.println("  /panel              doc ma chip dieu khien man hinh (ILI9341/ST7789)");
    Serial.println("  /drv st7789|ili9341|auto   doi driver panel (can /reboot)");
    Serial.println("  /inv on|off         dao mau / tat dao mau (INVON/INVOFF)");
    Serial.println("  /rgb rgb|bgr        doi thu tu mau RGB <-> BGR");
    Serial.println("  /imgrot 0..3        xoay/lat HINH CAMERA tren CYD (0=thuong 1=180 2=lat ngang 3=lat doc)");
    Serial.println("  /touch              bat/tat in toa do cam ung");
    Serial.println("  /stats              thong so stream / heap / fps");
    Serial.println("  /reboot             khoi dong lai");
    return;
  }

  if (!strcmp(line, "/ip")) {
    Serial.printf("[CFG] IP cam: %s\n", camIp);
    return;
  }

  if (strncmp(line, "/ip ", 4) == 0) {
    char *ip = line + 4;
    while (*ip == ' ') ip++;
    if (isIp4Lan(ip)) {
      strncpy(camIp, ip, sizeof(camIp) - 1);
      camIp[sizeof(camIp) - 1] = '\0';
      prefs.putString("camip", camIp);
      resultHttpClose();
      streamStop();
      Serial.printf("[CFG] Dat IP cam: %s\n", camIp);
    } else {
      Serial.println("[CFG] IP khong hop le");
    }
    return;
  }

  if (!strcmp(line, "/find")) {
    Serial.println("[DISC] Tim camera qua UDP...");
    if (udpReady) discoverCam(1500);
    return;
  }

  if (!strcmp(line, "/scan")) {
    Serial.println("[CAM] /scan: ep mo stream 20s de test video (khong can bam nut)");
    forcedScan = true;
    lastScanNotifyMs = millis();
    startScan("ep tay /scan");
    lastRetry = 0;
    lastFrameValid = false;
    return;
  }

  if (!strcmp(line, "/wizard")) {
    runOrientationWizard();
    return;
  }

  if (!strcmp(line, "/panel")) {
    printPanelInfo();
    Serial.printf("[TFT] rotation %u -> %dx%d\n", (unsigned)g_rotation,
                  tft.width(), tft.height());
    return;
  }

  if (!strncmp(line, "/drv", 4) &&
      (line[4] == '\0' || line[4] == ' ')) {
    char *arg = line + 4;
    while (*arg == ' ') arg++;
    if (!strcmp(arg, "st7789")) {
      prefs.putUChar("drv", 1);
      Serial.println("[TFT] Da chon driver ST7789. Go /reboot de ap dung.");
    } else if (!strcmp(arg, "ili9341")) {
      prefs.putUChar("drv", 2);
      Serial.println("[TFT] Da chon driver ILI9341. Go /reboot de ap dung.");
    } else if (!strcmp(arg, "auto")) {
      prefs.putUChar("drv", 0);
      Serial.println("[TFT] Che do tu dong do panel. Go /reboot de ap dung.");
    } else {
      Serial.printf("[TFT] driver dang dat: drv=%u (0=tu dong, 1=ST7789, 2=ILI9341)\n",
                    (unsigned)prefs.getUChar("drv", 1));
      Serial.println("[TFT] Dung: /drv st7789 | /drv ili9341 | /drv auto");
    }
    return;
  }

  // Dao mau (INVON/INVOFF) - doi ngay, luu NVS
  if (!strncmp(line, "/inv", 4) && (line[4] == '\0' || line[4] == ' ')) {
    char *arg = line + 4;
    while (*arg == ' ') arg++;
    if (!strcmp(arg, "on")) {
      g_inv = 1; prefs.putUChar("inv", 1); applyInversion(true);
    } else if (!strcmp(arg, "off")) {
      g_inv = 0; prefs.putUChar("inv", 0); applyInversion(false);
    } else {
      Serial.printf("[TFT] Dao mau hien tai: %s. Dung: /inv on | /inv off\n",
                    g_inv ? "BAT (INVON)" : "TAT (INVOFF)");
    }
    return;
  }

  // Xoay/lat HINH CAMERA chi tren CYD (khong doi gi ben ESP32-CAM)
  if (!strncmp(line, "/imgrot", 7) && (line[7] == '\0' || line[7] == ' ')) {
    char *arg = line + 7;
    while (*arg == ' ') arg++;
    if (*arg == '\0') {
      Serial.printf("[IMG] Dang dung: %u = %s\n",
                    (unsigned)g_imgRot, imgRotName(g_imgRot));
      Serial.println("[IMG] Dung: /imgrot 0 = giu nguyen | 1 = xoay 180 | 2 = lat ngang | 3 = lat doc");
      return;
    }
    const uint8_t r = (uint8_t)constrain(atoi(arg), 0, 3);
    g_imgRot = r;
    prefs.putUChar("irot", r);
    Serial.printf("[IMG] Da luu: %u = %s (chi tren CYD, CAM khong doi)\n",
                  (unsigned)r, imgRotName(r));
    // Phan hoi NGAY tren man hinh: ve lai khung hinh dang giu (neu co), neu
    // khong co thi hien trang thong bao che do dang chon.
    if (lastFrameBytes > 0 && showJpeg(lastFrameBytes)) {
      drawOverlay();
    } else {
      char b[48];
      snprintf(b, sizeof(b), "%u = %s", (unsigned)r, imgRotName(r));
      drawStatusPage("XOAY HINH CAMERA", b, TFT_CYAN);
      standbyShown = false;
    }
    return;
  }

  // Thu tu mau RGB/BGR - doi ngay, luu NVS
  if (!strncmp(line, "/rgb", 4) && (line[4] == '\0' || line[4] == ' ')) {
    char *arg = line + 4;
    while (*arg == ' ') arg++;
    if (!strcmp(arg, "rgb")) {
      g_bgr = 0; prefs.putUChar("bgr", 0); writeMadctlColor();
      Serial.println("[TFT] Thu tu mau: RGB");
    } else if (!strcmp(arg, "bgr")) {
      g_bgr = 1; prefs.putUChar("bgr", 1); writeMadctlColor();
      Serial.println("[TFT] Thu tu mau: BGR");
    } else {
      Serial.printf("[TFT] Thu tu mau hien tai: %s. Dung: /rgb rgb | /rgb bgr\n",
                    g_bgr ? "BGR" : "RGB");
    }
    return;
  }

  if (!strcmp(line, "/test")) {
    Serial.println("[TFT] Chay self-test man hinh...");
    drawSelfTest();
    lastFrameValid = false;
    if (!scanning) {
      drawStandbyPage();
      standbyShown = true;
    }
    return;
  }

  if (strncmp(line, "/rot", 4) == 0 &&
      (line[4] == '\0' || line[4] == ' ' || line[4] == '\t')) {
    if (line[4] == '\0') {                 // /rot  -> xem huong dang dung
      Serial.printf("[TFT] rotation hien tai: %u (%dx%d). Dung: /rot 0..7\n",
                    (unsigned)g_rotation, tft.width(), tft.height());
      return;
    }
    const uint8_t prev = g_rotation;
    const uint8_t r = (uint8_t)(atoi(line + 4) & 7);
    applyRotation(r);

    // UI duoc thiet ke 320x240 NGANG. Rotation 0/2/4/6 la DOC 240x320 ->
    // UI se khong kin man hinh (thua 1/4 phia tren, hien dai nhieu).
    if (tft.width() < tft.height()) {
      Serial.printf("[TFT] BO QUA rotation %u: la DOC %dx%d, UI 320x240 se khong kin man hinh.\n",
                    (unsigned)r, tft.width(), tft.height());
      Serial.println("[TFT] Chi dung /rot 1, 3, 5 hoac 7 (NGANG 320x240).");
      applyRotation(prev);                 // quay lai huong ngang dang dung
      return;
    }

    prefs.putUChar("rot", r);              // Nho huong nay cho cac lan boot sau
    tft.fillScreen(TFT_BLACK);
    lastFrameValid = false;
    Serial.printf("[TFT] Da luu rotation %u -> %dx%d\n",
                  (unsigned)g_rotation, tft.width(), tft.height());
    if (!scanning) {
      char b[48];
      snprintf(b, sizeof(b), "rot %u: %dx%d (da luu)", (unsigned)g_rotation,
               tft.width(), tft.height());
      drawStatusPage("XOAY MAN HINH", b, TFT_CYAN);
    }
    return;
  }

  if (!strncmp(line, "/live", 5)) {
    char *arg = line + 5;
    while (*arg == ' ') arg++;
    if (!strcmp(arg, "on")) {
      manualPaused = false;
      idleDirty = true;
      standbyShown = false;
      lastRetry = 0;
      Serial.println("[UI] Bat lai xem truc tiep");
    } else if (!strcmp(arg, "off")) {
      manualPaused = true;
      streamStop();
      ss = SS_OFF;
      idleDirty = true;
      Serial.println("[UI] Tam dung xem truc tiep");
    } else {
      Serial.println("[UI] Dung: /live on|off");
    }
    return;
  }

  if (!strcmp(line, "/touch")) {
    Serial.println("[TOUCH] Debug cam ung da bat/tat");
    return;
  }

  if (!strcmp(line, "/stats")) {
    printStats();
    return;
  }

  if (!strcmp(line, "/reboot")) {
    Serial.println("[SYS] Reboot...");
    delay(100);
    ESP.restart();
    return;
  }

  Serial.printf("[LENH] Khong ro: %s\n", line);
}

static void serialPoll() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen > 0) {
        rxLine[rxLen] = '\0';
        handleSerialLine(rxLine);
        rxLen = 0;
      }
    } else if (rxLen < sizeof(rxLine) - 1) {
      rxLine[rxLen++] = c;
    }
  }
}

// Cam ung: hien chi de DO/kiem tra toa do (xem /touch). Da BO nut TAM DUNG
// tren man hinh vi cham nham se lam CYD khong mo stream -> mat hinh camera.
static void touchPoll() {
  if (!touch.touched()) return;

  uint16_t rx, ry;
  uint8_t rz;
  touch.readData(&rx, &ry, &rz);
  if (rz == 0) return;

  uint16_t mx = rx;
  uint16_t my = ry;
#if TOUCH_SWAP_XY
  uint16_t tmp = mx; mx = my; my = tmp;
#endif
  long x = map((long)mx, TOUCH_X_MIN, TOUCH_X_MAX, 0, LCD_W - 1);
  long y = map((long)my, TOUCH_Y_MIN, TOUCH_Y_MAX, 0, LCD_H - 1);
#if TOUCH_INVERT_X
  x = (LCD_W - 1) - x;
#endif
#if TOUCH_INVERT_Y
  y = (LCD_H - 1) - y;
#endif
  x = constrain(x, 0, LCD_W - 1);
  y = constrain(y, 0, LCD_H - 1);

  Serial.printf("[TOUCH] raw %u,%u -> %ld,%ld\n",
                (unsigned)mx, (unsigned)my, x, y);
}

// ============================== PANEL ID / DRIVER ===========================
// Doc ma chip dieu khien bang lenh RDDID (0x04) qua SPI truc tiep.
//   ILI9341 -> 0x009341     ST7789 -> 0x858552 (hoac 0x8585xx)
// Tra ve 0 neu khong doc duoc (nhieu board khong noi MISO cua man hinh).
static uint32_t readPanelId() {
#if defined(TFT_MISO) && (TFT_MISO >= 0)
  SPIClass &spi = tft.getSPIinstance();

  pinMode(TFT_CS, OUTPUT);
  pinMode(TFT_DC, OUTPUT);
  pinMode(TFT_MISO, INPUT_PULLUP);
  digitalWrite(TFT_CS, HIGH);

  spi.beginTransaction(SPISettings(4000000, MSBFIRST, TFT_SPI_MODE));
  digitalWrite(TFT_CS, LOW);
  digitalWrite(TFT_DC, LOW);          // lenh
  spi.transfer(0x04);                 // RDDID
  digitalWrite(TFT_DC, HIGH);         // du lieu
  spi.transfer(0x00);                 // dummy
  const uint8_t a = spi.transfer(0x00);
  const uint8_t b = spi.transfer(0x00);
  const uint8_t c = spi.transfer(0x00);
  digitalWrite(TFT_CS, HIGH);
  spi.endTransaction();

  return ((uint32_t)a << 16) | ((uint32_t)b << 8) | c;
#else
  return 0;
#endif
}

static bool looksLikeST7789(uint32_t id) {
  if (!id) return false;
  if ((id >> 8) == 0x8585) return true;          // 0x8585xx
  if ((id >> 8) == 0x7789 || (id & 0xFFFF) == 0x7789) return true;
  return false;
}

static bool looksLikeILI9341(uint32_t id) {
  return (id >> 8) == 0x0093 || (id & 0xFFFF) == 0x9341;
}

// Nap lai chuoi init cua ST7789 (dung khi panel thuc te la ST7789 nhung
// TFT_eSPI dang duoc build voi ILI9341_DRIVER).
static void initPanelST7789() {
  static const uint8_t gamP[14] = {0xD0, 0x00, 0x02, 0x07, 0x0A, 0x28, 0x32,
                                   0x44, 0x42, 0x06, 0x0E, 0x12, 0x14, 0x17};
  static const uint8_t gamN[14] = {0xD0, 0x00, 0x02, 0x07, 0x0A, 0x28, 0x31,
                                   0x54, 0x47, 0x0E, 0x1C, 0x17, 0x1B, 0x1E};

  tft.writecommand(0x01);                        // SWRESET
  delay(150);
  tft.writecommand(0x11);                        // SLPOUT
  delay(120);
  tft.writecommand(0x13);                        // NORON

  tft.writecommand(0x36); tft.writedata(0x00);   // MADCTL (setRotation lo lai sau)
  tft.writecommand(0xB6); tft.writedata(0x0A); tft.writedata(0x82);
  tft.writecommand(0xB0); tft.writedata(0x00); tft.writedata(0xE0);
  tft.writecommand(0x3A); tft.writedata(0x55);   // 16-bit mau
  tft.writecommand(0xB2); tft.writedata(0x0C); tft.writedata(0x0C);
  tft.writedata(0x00); tft.writedata(0x33); tft.writedata(0x33);
  tft.writecommand(0xB7); tft.writedata(0x35);
  tft.writecommand(0xBB); tft.writedata(0x28);
  tft.writecommand(0xC0); tft.writedata(0x0C);
  tft.writecommand(0xC2); tft.writedata(0x01); tft.writedata(0xFF);
  tft.writecommand(0xC3); tft.writedata(0x10);
  tft.writecommand(0xC4); tft.writedata(0x20);
  tft.writecommand(0xC6); tft.writedata(0x0F);
  tft.writecommand(0xD0); tft.writedata(0xA4); tft.writedata(0xA1);
  tft.writecommand(0xE0);
  for (uint8_t i = 0; i < sizeof(gamP); ++i) tft.writedata(gamP[i]);
  tft.writecommand(0xE1);
  for (uint8_t i = 0; i < sizeof(gamN); ++i) tft.writedata(gamN[i]);
  // Dao mau (INVON/INVOFF) do ham applyInversion() quyet dinh theo NVS,
  // vi panel CYD ST7789 can INVOFF con mot so panel khac can INVON.
  tft.writecommand(0x29);                        // DISPON
  delay(50);
}

// Bat/tat dao mau (0x21 = INVON, 0x20 = INVOFF). CYD ST7789 can INVOFF.
static void applyInversion(bool on) {
  tft.writecommand(on ? 0x21 : 0x20);
  Serial.printf("[TFT] Dao mau: %s\n", on ? "BAT (INVON)" : "TAT (INVOFF)");
}

// In chan doan panel ra Serial (dung o setup va lenh /panel).
static void printPanelInfo() {
  const uint32_t id = readPanelId();
  Serial.printf("[TFT] Panel ID (SPI truc tiep, RDDID 0x04) = 0x%06X\n",
                (unsigned)id);
  if (looksLikeST7789(id))      Serial.println("[TFT]   -> nhan dien: ST7789");
  else if (looksLikeILI9341(id)) Serial.println("[TFT]   -> nhan dien: ILI9341");
  else if (id)                   Serial.println("[TFT]   -> khong ro (ma la)");
  else                           Serial.println("[TFT]   -> khong doc duoc (MISO?)");

  // Thu them co che doc indexed cua TFT_eSPI (chi ILI9341 moi ho tro)
  const uint8_t r04a = tft.readcommand8(0x04, 0);
  const uint8_t r04b = tft.readcommand8(0x04, 1);
  const uint8_t r04c = tft.readcommand8(0x04, 2);
  Serial.printf("[TFT] readcommand8 0x04 = %02X %02X %02X\n",
                r04a, r04b, r04c);
}

// ============================== RAM =========================================
// Cap bo dem JPEG: uu tien PSRAM; neu khong co thi tu tut kich thuoc dan cho
// toi khi vua DRAM (tranh truong hop cap 112KB that bai -> khong co bo dem ->
// khong bao gio hien duoc video).
static void allocJpegBuffer() {
  static const uint16_t kSizesKB[] = {112, 96, 80, 64, 48, 32, 24, 16};

  for (uint8_t i = 0; i < sizeof(kSizesKB) / sizeof(kSizesKB[0]); ++i) {
    const size_t bytes = (size_t)kSizesKB[i] * 1024;

    // 1. Thu PSRAM (board co PSRAM)
    uint8_t *p = (uint8_t *)heap_caps_malloc(bytes,
                                             MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    // 2. Khong co PSRAM (hoac PSRAM het) -> thu DRAM neu con khoi lien tuc du
    if (!p) {
      const size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
      if (largest > bytes + 8192) p = (uint8_t *)malloc(bytes);
    }

    if (p) {
      jpegBuf = p;
      jpegCap = bytes;
      Serial.printf("[RAM] JPEG buffer: %u KB (heap trong: %u B, lon nhat %u B)\n",
                    (unsigned)kSizesKB[i], (unsigned)ESP.getFreeHeap(),
                    (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
      return;
    }
  }

  jpegBuf = nullptr;
  jpegCap = 0;
  Serial.printf("[RAM] LOI: khong cap duoc bo dem JPEG (heap %u B) -> khong xem duoc video!\n",
                (unsigned)ESP.getFreeHeap());
}

// ============================== SETUP =======================================
void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("=== CYD ESP32-2432S028R - MAN HINH CAN MAT ===");
  Serial.printf("=== Firmware %s ===\n", FW_VERSION);

  // NVS: noi luu IP camera + rotation da chon
  prefs.begin("cyd", false);

  allocJpegBuffer();

  // Backlight truoc khi init de tranh nhay trang
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, TFT_BACKLIGHT_ON);

  tft.init();
  tft.setSwapBytes(true);                  // JPEGDEC xuat RGB565 little-endian

  // Doc ma chip dieu khien. Neu la ST7789 (mot so ban CYD khong "R") thi nap
  // lai chuoi init ST7789 vi TFT_eSPI dang duoc build voi ILI9341_DRIVER.
  printPanelInfo();
  // Che do mau (thu tu mau + dao mau) da luu bang /rgb va /inv
  // Mac dinh cho board CYD ST7789 cua du an: BGR + INVOFF (da kiem chung).
  g_bgr = prefs.getUChar("bgr", 1);
  g_inv = prefs.getUChar("inv", 0);

  // Xoay/lat hinh camera tren CYD (mac dinh 1 = xoay 180 do)
  g_imgRot = prefs.getUChar("irot", 1);
  if (g_imgRot > 3) g_imgRot = 1;
  Serial.printf("[IMG] Hinh camera tren CYD: %u = %s (/imgrot 0..3)\n",
                (unsigned)g_imgRot, imgRotName(g_imgRot));

  {
    // drv: 0 = tu dong do panel, 1 = ep ST7789, 2 = ep ILI9341 (khong lam gi)
    const uint32_t pid = readPanelId();
    // Mac dinh 1 = ST7789 vi board cua du an nay dung panel ST7789V
    // (board 2 cong USB, ESP32-2432S028 khong "R"). Doi bang /drv.
    const uint8_t drv = prefs.getUChar("drv", 1);
    const bool useST = (drv == 1) || (drv == 0 && looksLikeST7789(pid));

    if (useST) {
      Serial.println("[TFT] Nap chuoi init ST7789...");
      initPanelST7789();
      tft.setSwapBytes(true);
      // Panel CYD ST7789 can INVOFF (0x20) -> khong dao mau
      applyInversion(g_inv != 0);
      Serial.println("[TFT] Da nap xong init ST7789");
    } else {
      Serial.printf("[TFT] Giu init ILI9341 (drv=%u)\n", (unsigned)drv);
      if (prefs.isKey("inv")) applyInversion(g_inv != 0);
    }
  }

  // Khi firmware doi phien ban, bo qua rotation cu (co the chon sai o ban
  // truoc) de ve mac dinh 1 = landscape 320x240.
  if (prefs.getString("rotfw", "") != FW_VERSION) {
    prefs.putString("rotfw", FW_VERSION);
    // Chi xoa khi rotation cu la DOC (0/2/4/6) - huong NGANG (1/3/5/7) giu lai
    const uint8_t curRot = prefs.getUChar("rot", 1);
    if (curRot > 7 || curRot == 0 || curRot == 2 || curRot == 4 || curRot == 6) {
      prefs.putUChar("rot", 1);
    }
  }

  // Rotation: lay cai da luu bang lenh /rot (nho qua cac lan reboot).
  const bool hadSavedRot = prefs.isKey("rot");
  uint8_t startRot = prefs.getUChar("rot", 5);   // 5 = NGANG, da kiem chung
  if (startRot > 7) startRot = 5;
  applyRotation(startRot);

  // UI la 320x240 NGANG. Neu rotation da luu la DOC (0/2/4/6) thi tu doi
  // sang 1 de man hinh khong bi thua 1/4 phia tren.
  if (tft.width() < tft.height()) {
    Serial.printf("[TFT] Rotation da luu (%u) la DOC -> tu doi sang 1 cho NGANG\n",
                  (unsigned)startRot);
    startRot = 1;
    applyRotation(1);
    prefs.putUChar("rot", 1);
  }

  Serial.printf("[TFT] Kich thuoc theo rotation: 0=%dx%d 1=%dx%d 2=%dx%d 3=%dx%d\n",
                TFT_WIDTH, TFT_HEIGHT, TFT_HEIGHT, TFT_WIDTH,
                TFT_WIDTH, TFT_HEIGHT, TFT_HEIGHT, TFT_WIDTH);

  // Neu nguoi dung CHUA chon rotation nao ma cai mac dinh khong ra landscape
  // thi tu thu 0..3 va chon cai landscape dau tien.
  if (!hadSavedRot && tft.width() <= tft.height()) {
    Serial.println("[TFT] CANH BAO: rotation mac dinh khong ra landscape -> thu 0..3");
    for (uint8_t r = 0; r < 4; r++) {
      tft.setRotation(r);
      Serial.printf("[TFT]   thu rot %u: %dx%d\n", (unsigned)r,
                    tft.width(), tft.height());
      if (tft.width() > tft.height()) { g_rotation = r; break; }
    }
    tft.setRotation(g_rotation);
    Serial.printf("[TFT] Chon rotation %u -> %dx%d\n",
                  (unsigned)g_rotation, tft.width(), tft.height());
  }

  if (hadSavedRot) {
    Serial.printf("[TFT] Dung rotation da luu (/rot %u) -> %dx%d\n",
                  (unsigned)g_rotation, tft.width(), tft.height());
  }

  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(1);
  Serial.printf("[TFT] driver %s native %dx%d -> rotation%u %dx%d (UI %dx%d)\n",
#if CYD_PANEL_ST7789
                "ST7789",
#else
                "ILI9341",
#endif
                TFT_WIDTH, TFT_HEIGHT, (unsigned)g_rotation,
                tft.width(), tft.height(), LCD_W, LCD_H);
  if (tft.width() != LCD_W || tft.height() != LCD_H) {
    Serial.println("[TFT] CHU Y: kich thuoc sau xoay KHAC 320x240 (UI se bi cat).");
    Serial.println("[TFT]        Go /rot 1 hoac /rot 3 de chon huong NGANG dung.");
  }

#if CYD_SELFTEST
  Serial.println("[TFT] Self-test panel (4 mau + vien + 4 goc)...");
  drawSelfTest();
  tft.fillScreen(TFT_BLACK);
#endif

  // Cam ung tren bus SPI rieng
  touchSPI.begin(TOUCH_SCLK_PIN, TOUCH_MISO_PIN, TOUCH_MOSI_PIN, TOUCH_CS_PIN);
  touch.begin(touchSPI);
  touch.setRotation(1);
  Serial.println("[TOUCH] XPT2046 san sang");

  drawSplash("Dang khoi dong...");

  // NVS: lay IP camera da luu (prefs da begin tu dau setup)
  String saved = prefs.getString("camip", "");
  if (saved.length() && isIp4Lan(saved.c_str())) {
    strncpy(camIp, saved.c_str(), sizeof(camIp) - 1);
    camIp[sizeof(camIp) - 1] = '\0';
  }
  Serial.printf("[CFG] IP cam: %s (mac dinh %s)\n", camIp, CAM_IP_DEFAULT);

  drawSplash("Dang ket noi WiFi...");
  if (!wifiConnect(20000)) {
    wifiDown = true;
    drawStatusPage("LOI WIFI", "Kiem tra SSID / mat khau", TFT_RED);
  } else {
    udpReady = (udp.begin(DISC_PORT) == 1);
    if (!udpReady) Serial.println("[DISC] Khong mo duoc UDP 4321");

    // Thu tim camera neu IP mac dinh khong dung
    if (!pollResult()) {
      Serial.println("[CAM] Chua doc duoc /result -> discovery");
      if (udpReady) discoverCam(800);
    }

    if (scanning) {
      drawStatusPage("DANG QUET", "Dang lay hinh tu camera...", TFT_ORANGE);
    } else {
      drawStandbyPage();   // nghi: khong truyen anh
      standbyShown = true;
    }
  }

  lastResultPoll = millis();
  lastFpsMs = millis();
  lastStatsMs = millis();
  lastRetry = 0;
  lastDiscovery = millis();
  Serial.println("[SYS] San sang. Go /help de xem lenh.");
}

// ============================== LOOP ========================================
void loop() {
  uint32_t now = millis();
  serialPoll();
  touchPoll();

  // ---------------- WiFi ----------------
  if (WiFi.status() != WL_CONNECTED) {
    if (ss != SS_OFF) streamStop();
    if (!wifiDown) {
      wifiDown = true;
      resultHttpClose();
      drawStatusPage("MAT WIFI", "Dang ket noi lai...", TFT_RED);
      Serial.println("[WiFi] Mat ket noi");
    }
    if (now - lastWifiTry > 8000) {
      lastWifiTry = now;
      Serial.println("[WiFi] Thu ket noi lai...");
      if (!wifiConnect(12000)) WiFi.disconnect();
    }
    delay(20);
    return;
  }
  if (wifiDown) {
    wifiDown = false;
    resultHttpClose();
    lastRetry = 0;
    lastFrameValid = false;
    Serial.println("[WiFi] Da ket noi lai");
    if (scanning) {
      drawStatusPage("DANG QUET", "Dang lay hinh tu camera...", TFT_ORANGE);
    } else {
      drawStandbyPage();
      standbyShown = true;
    }
  }

  // ---------------- Thong bao quet tu ESP32-CAM (UDP) ----------------
  pollUdp();

  // ---------------- Doc trang thai ESP32-CAM (/result) ----------------
  // Khi DANG QUET: KHONG poll (hoac poll rat thua) de nhuong thoi gian doc
  // frame tu stream :81. Viec bat/tat quet duoc bao qua UDP + fail-safe.
  const uint32_t pollInterval = scanning ? 2000 : RESULT_POLL_MS;
  if (now - lastResultPoll >= pollInterval) {
    lastResultPoll = now;
    pollResult();
  }

  // ---------------- Khi NGHI: dinh ky tim lai camera neu IP sai ----------------
  // Truoc day CYD chi di tim camera trong luc dang quet, nen neu IP camera sai
  // thi khong bao gio mo duoc stream -> bam nut cung khong hien hinh.
  if (!scanning && !wifiDown && (now - lastDiscovery > 10000)) {
    lastDiscovery = now;
    if (!pollResult()) {
      // Chi in toi da 1 lan/phut: CAM khong chay server :80 la binh thuong khi
      // chi xem hinh, log lap lai se lam rong man hinh Serial.
      static uint32_t lastNoResultLog = 0;
      if (now - lastNoResultLog > 60000) {
        lastNoResultLog = now;
        Serial.printf("[DISC] Chua doc duoc /result cua %s -> tim camera qua UDP\n",
                      camIp);
      }
      if (udpReady && discoverCam(700)) {
        Serial.printf("[DISC] Da cap nhat camera: %s\n", camIp);
        pollResult();
      }
    }
  }

  // ---------------- Fail-safe: mat tin hieu "dang quet" ----------------
  if (scanning) {
    // SO SANH CO DAU la bat buoc: lastScanNotifyMs duoc dat bang millis() MOI
    // (tre hon `now` lay o dau vong lap vai ms), nen tru kieu unsigned se bi
    // underflow -> "now - last" thanh so cuc lon -> cat phim quet NGAY LAP TUC.
    // Day chinh la loi khien BAM NUT khong hien hinh (moi lan "timeout" sau
    // 2ms) trong khi /scan van chay vi co co forcedScan.
    const uint32_t last = lastScanNotifyMs ? lastScanNotifyMs : scanStartMs;
    const int32_t age = (int32_t)(millis() - last);
    const int32_t total = (int32_t)(millis() - scanStartMs);
    const bool lost = !forcedScan && (age > (int32_t)SCAN_NOTIFY_MS);
    if (lost || total > (int32_t)SCAN_MAX_MS) {
      endScan(forcedScan ? "het 20s (ep tay)" : "timeout");
    }
  }

  // ---------------- Stream / decode ----------------
  // CHI truyen anh khi camera dang quet (bam nut). Khi nghI dong stream de
  // ESP32-CAM khong chup lien tuc -> giam nhiet, tiet kiem dien.
  bool wantStream = scanning && !manualPaused;

  if (!wantStream) {
    if (ss == SS_LIVE || ss == SS_CONNECTING) {
      streamStop();
      ss = SS_OFF;
    }
    if (manualPaused) {
      if (idleDirty) {
        drawIdlePage();
        idleDirty = false;
      }
    } else {
      // Vua quet xong: giu khung cuoi them RESULT_HOLD_MS roi ve trang cho.
      const bool holding = scanEndMs &&
          ((int32_t)(now - scanEndMs) < (int32_t)RESULT_HOLD_MS);
      if (!holding && !standbyShown) {
        drawStandbyPage();
        standbyShown = true;
        lastFrameValid = false;
      }
    }
  } else {
    if (ss == SS_LIVE) {
      size_t len = 0;
      int r = readFrame(&len);
      if (r == 1) {
        lastFrameBytes = len;
        if (showJpeg(len)) {
          framesThisSec++;
          framesTotal++;
          lastFrameValid = true;
          drawOverlay();
          if (framesTotal <= 3 || (framesTotal % 60) == 0) {
            Serial.printf("[%lums] [CAM] frame #%lu: %u B, decode %u ms\n",
                          (unsigned long)now, (unsigned long)framesTotal,
                          (unsigned)len, (unsigned)lastDecodeMs);
          }
        }
      } else if (r < 0) {
        Serial.printf("[%lums] [CAM] Stream loi -> ket noi lai\n",
                      (unsigned long)now);
        streamStop();
        ss = SS_RETRY;
        lastRetry = now;
        if (lastFrameValid) drawConnBox("MAT KET NOI", "Dang thu lai...");
        else drawStatusPage("DANG QUET", "Dang lay hinh tu camera...", TFT_ORANGE);
      }
      // r == 0: frame qua lon, da bo -> doc tiep frame sau

      // Canh bao: da ket noi ma 3s chua co frame nao
      static uint32_t tLiveWarn = 0;
      if (!lastFrameValid && now - tLiveWarn > 3000) {
        tLiveWarn = now;
        Serial.printf("[%lums] [CAM] Da ket noi nhung CHUA co frame nao (3s)\n",
                      (unsigned long)now);
      }
    } else if (ss == SS_RETRY || ss == SS_OFF) {
      if (now - lastRetry >= STREAM_RETRY_MS) {
        lastRetry = now;
        ss = SS_CONNECTING;
        if (!lastFrameValid) {
          drawStatusPage("DANG QUET", "Dang lay hinh tu camera...", TFT_ORANGE);
        }
        if (streamConnect()) {
          ss = SS_LIVE;
          lastDiscovery = now;
        } else {
          ss = SS_RETRY;
          if (!lastFrameValid) {
            drawStatusPage("KHONG THAY CAMERA", "Kiem tra nguon / WiFi cua cam", TFT_RED);
          }
          if (now - lastDiscovery > 12000) {
            lastDiscovery = now;
            if (udpReady) discoverCam(700);
          }
        }
      }
    }
  }

  // ---------------- FPS ----------------
  if (now - lastFpsMs >= 1000) {
    fps = framesThisSec;
    framesThisSec = 0;
    lastFpsMs = now;
  }

  // ---------------- Log dinh ky ----------------
  if (now - lastStatsMs >= 15000) {
    lastStatsMs = now;
    Serial.printf("[STATS] %u fps, frame %u B, decode %u ms, heap %u, ss=%d\n",
                  (unsigned)fps, (unsigned)lastFrameBytes,
                  (unsigned)lastDecodeMs, ESP.getFreeHeap(), (int)ss);
  }

  delay(1);
}

// Draw a compact connection warning over the last camera frame.
static void drawConnBox(const char *title, const char *sub) {
  tft.fillRoundRect(24, 78, 272, 86, 12, TFT_BLACK);
  tft.drawRoundRect(24, 78, 272, 86, 12, TFT_RED);
  drawCenteredFit(title, LCD_W / 2, 109, 244, 2, TFT_RED, TFT_BLACK);
  drawCenteredFit(sub, LCD_W / 2, 138, 244, 1, TFT_LIGHTGREY, TFT_BLACK);
}
