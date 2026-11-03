
//  CAU HINH PERSISTENT (NVS) - xem app_config.h
//  Luu tat ca trong Preferences namespace "camcfg". Chi doc/ghi cac key can
//  thiet -> NVS khong bi phinh, va co the them/sua key sau nay ma khong mat
//  du lieu cu.

#include "app_config.h"
#include <Preferences.h>
#include <WiFi.h>

CamCfg cfg;
bool cfg_setup_mode = false;

static Preferences prefs;
static const char *NS = "camcfg";

// Mac dinh truyen vao cfg_load() -> giu lai de cfg_reset() dung khi khoi phuc
static char def_ssid_buf[33] = "";
static char def_pass_buf[65] = "";
static char def_push_buf[128] = "";


//  Mac dinh: truyen tu .ino (ssid/password/PUSH_URL dang hard-code) -> neu
//  chua luu gi trong NVS thi firmware van hoat dong nhu truoc day.

static void cfg_defaults(const char *def_ssid, const char *def_pass, const char *def_push) {
  // Chi ghi bo dem mac dinh khi khac con tro (cfg_reset() truyen chinh no)
  if (def_ssid != def_ssid_buf) strlcpy(def_ssid_buf, def_ssid ? def_ssid : "", sizeof(def_ssid_buf));
  if (def_pass != def_pass_buf) strlcpy(def_pass_buf, def_pass ? def_pass : "", sizeof(def_pass_buf));
  if (def_push != def_push_buf) strlcpy(def_push_buf, def_push ? def_push : "", sizeof(def_push_buf));

  memset(&cfg, 0, sizeof(cfg));
  strlcpy(cfg.wifi_ssid, def_ssid_buf, sizeof(cfg.wifi_ssid));
  strlcpy(cfg.wifi_pass, def_pass_buf, sizeof(cfg.wifi_pass));

  cfg.use_static_ip = 0;
  strlcpy(cfg.static_ip, "192.168.0.100", sizeof(cfg.static_ip));
  strlcpy(cfg.static_gw, "192.168.0.1", sizeof(cfg.static_gw));
  strlcpy(cfg.static_mask, "255.255.255.0", sizeof(cfg.static_mask));
  strlcpy(cfg.static_dns, "8.8.8.8", sizeof(cfg.static_dns));

  strlcpy(cfg.push_url, def_push_buf, sizeof(cfg.push_url));

  cfg.vflip = -1;      // mac dinh theo cam bien (OV3660 se bi lat nguoc -> xem camWakeInit)
  cfg.hmirror = -1;

  strlcpy(cfg.ap_ssid, "ESP32-CAM-Setup", sizeof(cfg.ap_ssid));
  strlcpy(cfg.ap_pass, "12345678", sizeof(cfg.ap_pass));  // WPA2 can >= 8 ky tu
}

void cfg_load(const char *def_ssid, const char *def_pass, const char *def_push) {
  cfg_defaults(def_ssid, def_pass, def_push);

  prefs.begin(NS, true);   // read-only
  if (prefs.isKey("ssid")) {   // co luu roi -> doc tung key (thieu key thi giu mac dinh)
    prefs.getString("ssid", cfg.wifi_ssid, sizeof(cfg.wifi_ssid));
    prefs.getString("pass", cfg.wifi_pass, sizeof(cfg.wifi_pass));
    cfg.use_static_ip = prefs.getUChar("use_st", 0);
    prefs.getString("ip", cfg.static_ip, sizeof(cfg.static_ip));
    prefs.getString("gw", cfg.static_gw, sizeof(cfg.static_gw));
    prefs.getString("mask", cfg.static_mask, sizeof(cfg.static_mask));
    prefs.getString("dns", cfg.static_dns, sizeof(cfg.static_dns));
    prefs.getString("push", cfg.push_url, sizeof(cfg.push_url));
    cfg.vflip = (int8_t)prefs.getChar("vflip", (char)-1);
    cfg.hmirror = (int8_t)prefs.getChar("hmir", (char)-1);
    prefs.getString("ap_ssid", cfg.ap_ssid, sizeof(cfg.ap_ssid));
    prefs.getString("ap_pass", cfg.ap_pass, sizeof(cfg.ap_pass));
  }
  prefs.end();

  // Canh bao gia tri te -> tranh mat ket noi khi reboot
  if (cfg.push_url[0] == '\0') {
    strlcpy(cfg.push_url, def_push_buf, sizeof(cfg.push_url));
    Serial.println("[CFG] push_url trong NVS rong -> lay lai mac dinh tu .ino");
  }
  Serial.printf("[CFG] ssid='%s' static_ip=%d push='%s' vflip=%d hmirror=%d ap='%s'\n",
                cfg.wifi_ssid, cfg.use_static_ip, cfg.push_url, cfg.vflip, cfg.hmirror, cfg.ap_ssid);
}

void cfg_save() {
  prefs.begin(NS, false);
  prefs.putString("ssid", cfg.wifi_ssid);
  prefs.putString("pass", cfg.wifi_pass);
  prefs.putUChar("use_st", cfg.use_static_ip);
  prefs.putString("ip", cfg.static_ip);
  prefs.putString("gw", cfg.static_gw);
  prefs.putString("mask", cfg.static_mask);
  prefs.putString("dns", cfg.static_dns);
  prefs.putString("push", cfg.push_url);
  prefs.putChar("vflip", (char)cfg.vflip);
  prefs.putChar("hmir", (char)cfg.hmirror);
  prefs.putString("ap_ssid", cfg.ap_ssid);
  prefs.putString("ap_pass", cfg.ap_pass);
  prefs.end();
  Serial.println("[CFG] da luu cau hinh xuong NVS");
}

void cfg_reset() {
  prefs.begin(NS, false);
  prefs.clear();
  prefs.end();
  cfg_defaults(def_ssid_buf, def_pass_buf, def_push_buf);
  Serial.println("[CFG] da khoi phuc cau hinh mac dinh");
}


//  IP tinh: WiFi.config(local_ip, gateway, subnet, dns) PHAI goi TRUOC
//  WiFi.begin() (sau do DHCP se bi bo qua). Tra ve false neu chua dung
//  hoac IP nhap sai dinh dang.

bool cfg_apply_static_ip() {
  if (!cfg.use_static_ip) return false;
  IPAddress ip, gw, mask, dns;
  if (!ip.fromString(cfg.static_ip) || !gw.fromString(cfg.static_gw) || !mask.fromString(cfg.static_mask)) {
    Serial.println("[CFG] IP tinh sai dinh dang (vd 192.168.1.100) -> bo qua, dung DHCP");
    return false;
  }
  if (!dns.fromString(cfg.static_dns)) dns = IPAddress(8, 8, 8, 8);
  bool ok = WiFi.config(ip, gw, mask, dns);
  Serial.printf("[CFG] IP tinh %s (%s, gw %s, mask %s, dns %s)\n",
                ok ? "da cai" : "THAT BAI", ip.toString().c_str(),
                gw.toString().c_str(), mask.toString().c_str(), dns.toString().c_str());
  return ok;
}
