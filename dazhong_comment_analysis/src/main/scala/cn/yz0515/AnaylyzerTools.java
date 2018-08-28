package cn.yz0515;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.lionsoul.jcseg.tokenizer.ASegment;
import org.lionsoul.jcseg.tokenizer.core.*;

/**
 * 分词工具
 * 参考： 基于Spark上的中文分词算法的实现 https://blog.csdn.net/nuoyahadili8/article/details/51088356
 */
public class AnaylyzerTools {
    public static ArrayList<String> anaylyzerWords(String str) {
        JcsegTaskConfig config = new JcsegTaskConfig(AnaylyzerTools.class.getResource("").getPath()+"jcseg.properties");
        ADictionary dic = DictionaryFactory.createDefaultDictionary(config);
        ArrayList<String> list = new ArrayList<String>();
        ASegment seg = null;
        try {
            seg = (ASegment) SegmentFactory.createJcseg(JcsegTaskConfig.COMPLEX_MODE, new Object[]{config, dic});
        } catch (JcsegException e1) {
            e1.printStackTrace();
        }
        try {
            seg.reset(new StringReader(str));
            IWord word = null;
            while ( (word = seg.next()) != null ) {
                list.add(word.getValue());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return list;
    }

    public static void main(String[] args) {
        String str = "HBase中通过row和columns确定的为一个存贮单元称为cell。显示每个元素，每个 cell都保存着同一份数据的多个版本。版本通过时间戳来索引。迎泽区是繁华的地方,营业厅营业";
        List<String> list = AnaylyzerTools.anaylyzerWords(str);
        for(String word:list) {
            System.out.println(word);
        }
        System.out.println(list.size());
    }
}
