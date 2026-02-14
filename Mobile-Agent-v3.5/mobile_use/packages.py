"""
Package name mapping between Android package identifiers and human-readable app names.
Provides bidirectional lookup: package -> names and name -> packages.
"""

PACKAGE_STR_LIST = '''com.tencent.mm	微信	wechat			
com.tencent.mobileqq	qq	腾讯qq			
com.sina.weibo	微博				
com.taobao.taobao	淘宝				
com.jingdong.app.mall	京东	京东秒送			
com.xunmeng.pinduoduo	拼多多				
com.xingin.xhs	小红书				
com.douban.frodo	豆瓣				
com.zhihu.android	知乎				
com.autonavi.minimap	高德地图	高德			
com.baidu.BaiduMap	百度地图				
com.sankuai.meituan.takeoutnew	美团外卖				
com.sankuai.meituan	美团	美团外卖			
com.dianping.v1	大众点评	点评			
me.ele	饿了么	淘宝闪购			
com.yek.android.kfc.activitys	肯德基				
ctrip.android.view	携程	携程旅行			
com.MobileTicket	铁路12306	12306			
com.Qunar	去哪儿旅行	去哪儿网	去哪儿		
com.sdu.didi.psnger	滴滴出行	滴滴			
tv.danmaku.bili	bilibili	b站	哔哩哔哩	哔站	bili
com.ss.android.ugc.aweme	抖音				
com.smile.gifmaker	快手				
com.tencent.qqlive	腾讯视频				
com.qiyi.video	爱奇艺				
com.youku.phone	优酷	优酷视频			
com.hunantv.imgo.activity	芒果tv	芒果			
com.phoenix.read	红果短剧	红果			
com.netease.cloudmusic	网易云音乐	网易云			
com.tencent.qqmusic	qq音乐				
com.luna.music	汽水音乐				
com.ximalaya.ting.android	喜马拉雅				
com.dragon.read	番茄免费小说	番茄小说			
com.kmxs.reader	七猫免费小说				
com.ss.android.lark	飞书				
com.tencent.androidqqmail	qq邮箱				
com.larus.nova	豆包	豆包			
com.gotokeep.keep	keep				
com.lingan.seeyou	美柚				
com.tencent.news	腾讯新闻				
com.ss.android.article.news	今日头条				
com.lianjia.beike	贝壳找房				
com.anjuke.android.app	安居客				
com.hexin.plat.android	同花顺				
com.miHoYo.hkrpg	星穹铁道	崩坏			
com.papegames.lysk.cn	恋与深空				
com.android.settings	settings	androidsystemsettings			
com.android.soundrecorder	audiorecorder				
com.rammigsoftware.bluecoins	bluecoins				
com.flauschcode.broccoli	broccoli				
com.booking	booking				
com.android.chrome	谷歌浏览器	googlechrome	chrome		
com.android.deskclock	时钟	闹钟	clock		
com.android.contacts	contacts				
com.duolingo	duolingo	多邻国			
com.expedia.bookings	expedia				
com.android.fileexplorer	files	filemanager			
com.google.android.gm	gmail	googlemail			
com.google.android.apps.nbu.files	googlefiles	filesbygoogle			
com.google.android.calendar	googlecalendar				
com.google.android.apps.dynamite	googlechat				
com.google.android.deskclock	googleclock				
com.google.android.contacts	googlecontacts				
com.google.android.apps.docs.editors.docs	googledocs				
com.google.android.apps.docs	googledrive				
com.google.android.apps.fitness	googlefit				
com.google.android.keep	googlekeep				
com.google.android.apps.maps	googlemaps				
com.google.android.apps.books	googleplaybooks				
com.android.vending	googleplaystore				
com.google.android.apps.docs.editors.slides	googleslides				
com.google.android.apps.tasks	googletasks				
net.cozic.joplin	joplin				
com.mcdonalds.app	麦当劳	mcdonald			
net.osmand	osmand				
com.Project100Pi.themusicplayer	pimusicplayer				
com.quora.android	quora				
com.reddit.frontpage	reddit				
code.name.monkey.retromusic	retromusic				
com.scientificcalculatorplus.simplecalculator.basiccalculator.mathcalc	simplecalendarpro				
com.simplemobiletools.smsmessenger	simplesmsmessenger				
org.telegram.messenger	telegram				
com.einnovation.temu	temu				
com.zhiliaoapp.musically	tiktok				
com.twitter.android	twitter	x			
org.videolan.vlc	vlc				
com.whatsapp	whatsapp				
com.taobao.movie.android	淘票票				
com.tongcheng.android	同程旅行	同程			
com.sankuai.movie	猫眼				
com.wuba.zhuanzhuan	转转				
com.tencent.weread	微信读书				
com.taobao.idlefish	闲鱼				
com.wudaokou.hippo	盒马				
com.eg.android.AlipayGphone	支付宝				
com.jd.jrapp	京东金融				
com.achievo.vipshop	唯品会				
com.smzdm.client.android	什么值得买				
cn.kuwo.player	酷我音乐				
com.taobao.trip	飞猪	飞猪旅行			
com.jingdong.pdj	京东到家				
com.tencent.map	腾讯地图				
com.shizhuang.duapp	得物				
cn.damai	大麦	大麦网			
com.ss.android.auto	懂车帝				
com.cubic.autohome	汽车之家				
com.wuba	58同城	五八同城			
com.android.calendar	日历				
com.alibaba.android.rimet	钉钉				
com.meituan.retail.v.android	小象超市				
com.aliyun.tongyi	通义	千问	通义千问		
com.hupu.games	虎扑	虎扑体育			
com.quark.browser	夸克	夸克浏览器			
com.yuantiku.tutor	猿辅导				
com.tencent.mtt	qq浏览器				
com.umetrip.android.msky.app	航旅纵横				
com.UCMobile	UC浏览器				
com.ss.android.ugc.aweme.lite	抖音极速版	抖音			
air.tv.douyu.android	斗鱼				
com.tencent.hunyuan.app.chat	元宝				
com.baidu.searchbox	百度				
com.lemon.lv	剪映				
cn.soulapp.android	soul				
com.baidu.netdisk	百度网盘				
com.tmri.app.main	交管12123	12123			
com.kugou.android	酷狗	酷狗音乐			
com.ss.android.lark	飞书				
com.tencent.android.qqdownloader	应用宝				
com.mt.mtxx.mtxx	美图	美图秀秀			
com.tencent.karaoke	全民k歌				
com.intsig.camscanner	扫描全能王				
com.android.bankabc	农业银行	农行			
cmb.pb	招商银行	招行			
com.ganji.android.haoche_c	瓜子二手车	瓜子			
com.sf.activity	顺丰	顺丰快递	顺丰速运		
com.ziroom.ziroomcustomer	自如				
com.yumc.phsuperapp	必胜客				
cn.dominos.pizza	达美乐披萨	达美乐			
cn.wps.moffice_eng	WPS Office	WPS			
com.mfw.roadbook	马蜂窝				
com.moonshot.kimichat	kimi				
com.tencent.wemeet.app	腾讯会议				
com.deepseek.chat	deepseek				
com.spdbccc.app	浦发银行				
cn.samsclub.app	山姆超市	山姆	山姆会员商店	山姆会员店	
com.tencent.qqsports	腾讯体育				
com.hanweb.android.zhejiang.activity	浙里办				
com.ss.android.article.video	西瓜视频				
com.taou.maimai	脉脉	'''


def normalize_package_name(name):
    """Normalize an app name by converting to lowercase and removing spaces/hyphens."""
    return name.lower().strip().replace(" ", "").replace("-", "")


def build_package_dicts():
    """
    Build bidirectional lookup dictionaries from the package string list.

    Returns:
        packages_name_dict: {package_id: [normalized_name1, normalized_name2, ...]}
        name_package_dict:  {normalized_name: [package_id1, package_id2, ...]}
    """
    packages_name_dict = {}
    name_package_dict = {}

    for line in PACKAGE_STR_LIST.strip().split("\n"):
        parts = line.strip().split("\t")
        if not parts:
            continue
        package_id = parts[0]
        names = [normalize_package_name(n) for n in parts[1:] if n.strip()]
        packages_name_dict[package_id] = names

        for name in names:
            if name not in name_package_dict:
                name_package_dict[name] = [package_id]
            else:
                name_package_dict[name].append(package_id)

    return packages_name_dict, name_package_dict


# Module-level singleton dictionaries
PACKAGES_NAME_DICT, NAME_PACKAGE_DICT = build_package_dicts()
