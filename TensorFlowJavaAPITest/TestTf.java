package TensorFlowJavaAPITest;

import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.SavedModelBundle;
//import org.tensorflow.SavedModelBundle;


import java.nio.FloatBuffer;
import java.util.Arrays;

public class TestTf {

   public static void main(String[] args) {
      SavedModelBundle smb = SavedModelBundle.load("export", "tag");
      Session s = smb.session();

      float[][] matrix={{1.0F,2.0F,3.0F,4.0F}};//[[1.0F,2.0F,3.0F,4.0F]] 0行4列

      System.out.println(Arrays.deepToString(matrix));

      Tensor xFeed= Tensor.create(matrix);
      Tensor result = s.runner().feed("x",xFeed).fetch("y").run().get(0);
      FloatBuffer buf =FloatBuffer.allocate(2);
      result.writeTo(buf);
      System.out.println(result.toString());
      System.out.println(buf.get(0));
      System.out.println(buf.get(1));
   }
}