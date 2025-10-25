diff --git a/README.md b/README.md
index 5cf657a5d488fef80c8ce85ae53c10c3ca105225..0805e767c9858f150bba9d73fb00c6fc69b65f13 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,16 @@
 # pa-stratejisi
-Price Action trading stratejisi
+
+Price Action ve pekiştirmeli öğrenme (RL) tabanlı trading botu için yapılandırma dosyası deposu.
+
+## Konfigürasyon
+
+`config/pa_rl_trading_bot.yaml` dosyası, botun risk yönetimi, işlem filtreleri, yeniden giriş kuralları ve RL ajanı için gerekli tüm parametreleri içerir. Dosya varsayılan olarak kâğıt (paper) modunda çalışacak şekilde ayarlanmıştır ve yorum satırlarında her bölümün hangi strateji parçasına ait olduğu belirtilmiştir.
+
+### Kullanım
+
+1. Dosyayı kendi projenize kopyalayın veya sembolik bağlantı oluşturun.
+2. `exchange` ve `telegram` bölümlerindeki kimlik bilgilerini kendi hesaplarınıza göre güncelleyin.
+3. Botu çalıştıran uygulamanın ilgili konfigürasyon yükleme mekanizmasına dosya yolunu tanıtın.
+4. Risk toleransınıza göre `risk` ve `behavior` bölümlerindeki eşikleri özelleştirin.
+
+Yapılandırmayı düzenlerken YAML biçiminin korunmasına dikkat edin ve yorum satırlarını gerektiğinde kendi notlarınızla güncelleyebilirsiniz.
