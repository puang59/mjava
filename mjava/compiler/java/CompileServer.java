import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;
import javax.tools.*;

/** Persistent in-memory javac server. One JSON request per stdin line, one JSON reply per stdout line. */
public class CompileServer {
    static final class Src extends SimpleJavaFileObject {
        final String code;
        Src(String name, String code) {
            super(URI.create("string:///" + name.replace('\\', '/')), Kind.SOURCE);
            this.code = code;
        }
        @Override public CharSequence getCharContent(boolean ignore) { return code; }
    }

    static final class Out extends SimpleJavaFileObject {
        Out(String name, Kind kind) { super(URI.create("mem:///" + name.replace('.', '/') + kind.extension), kind); }
        @Override public OutputStream openOutputStream() { return OutputStream.nullOutputStream(); }
    }

    static final class FM extends ForwardingJavaFileManager<JavaFileManager> {
        FM(JavaFileManager fm) { super(fm); }
        @Override public JavaFileObject getJavaFileForOutput(Location l, String cls, JavaFileObject.Kind k, FileObject sib) {
            return new Out(cls, k);
        }
    }

    // ---- minimal JSON object reader: flat object with string / integer values ----
    static Map<String, String> parse(String s) {
        Map<String, String> m = new HashMap<>();
        int i = 0, n = s.length();
        while (i < n && s.charAt(i) != '{') i++;
        i++;
        int[] pos = {0};
        while (i < n) {
            while (i < n && s.charAt(i) != '"' && s.charAt(i) != '}') i++;
            if (i >= n || s.charAt(i) == '}') break;
            pos[0] = i;
            String key = readString(s, pos);
            i = pos[0];
            while (i < n && s.charAt(i) != ':') i++;
            i++;
            while (i < n && Character.isWhitespace(s.charAt(i))) i++;
            if (s.charAt(i) == '"') {
                pos[0] = i;
                m.put(key, readString(s, pos));
                i = pos[0];
            } else {
                int j = i;
                while (j < n && s.charAt(j) != ',' && s.charAt(j) != '}') j++;
                m.put(key, s.substring(i, j).trim());
                i = j;
            }
        }
        return m;
    }

    static String readString(String s, int[] pos) {
        int i = pos[0] + 1;
        StringBuilder b = new StringBuilder();
        while (i < s.length()) {
            char c = s.charAt(i++);
            if (c == '"') break;
            if (c == (char) 92) {
                char e = s.charAt(i++);
                switch (e) {
                    case 'n': b.append('\n'); break;
                    case 't': b.append('\t'); break;
                    case 'r': b.append('\r'); break;
                    case 'b': b.append('\b'); break;
                    case 'f': b.append('\f'); break;
                    case 'u': b.append((char) Integer.parseInt(s.substring(i, i + 4), 16)); i += 4; break;
                    default: b.append(e);
                }
            } else b.append(c);
        }
        pos[0] = i;
        return b.toString();
    }

    static void esc(StringBuilder b, String s) {
        final char bs = (char) 92;
        b.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"') b.append(bs).append('"');
            else if (c == bs) b.append(bs).append(bs);
            else if (c == '\n') b.append(bs).append('n');
            else if (c == '\r') b.append(bs).append('r');
            else if (c == '\t') b.append(bs).append('t');
            else if (c < 0x20 || c > 0x7e) {
                b.append(bs).append('u');
                String h = Integer.toHexString(c);
                for (int k = h.length(); k < 4; k++) b.append('0');
                b.append(h);
            } else b.append(c);
        }
        b.append('"');
    }

    static String oneLine(String msg) {
        StringBuilder b = new StringBuilder();
        for (String ln : msg.split("[\r\n]+")) {
            String t = ln.trim();
            if (t.isEmpty() || t.startsWith("location:")) continue;
            if (b.length() > 0) b.append(' ');
            b.append(t);
        }
        return b.toString().replace("java.lang.", "").replaceAll("\\s+", " ");
    }

    public static void main(String[] args) throws Exception {
        JavaCompiler jc = ToolProvider.getSystemJavaCompiler();
        if (jc == null) { System.err.println("no system java compiler"); System.exit(2); }
        StandardJavaFileManager std = jc.getStandardFileManager(null, Locale.ENGLISH, StandardCharsets.UTF_8);
        FM fm = new FM(std);
        List<String> opts = List.of("-proc:none", "-Xlint:none");
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), false, StandardCharsets.UTF_8);
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) continue;
            StringBuilder r = new StringBuilder();
            String id = "-1";
            try {
                Map<String, String> req = parse(line);
                id = req.getOrDefault("id", "-1");
                if (req.containsKey("quit")) break;
                String fname = req.getOrDefault("filename", "Main.java");
                DiagnosticCollector<JavaFileObject> dc = new DiagnosticCollector<>();
                StringWriter sink = new StringWriter();
                boolean ok = jc.getTask(sink, fm, dc, opts, null,
                        List.of(new Src(fname, req.getOrDefault("source", "")))).call();
                r.append("{\"id\":").append(id).append(",\"ok\":").append(ok).append(",\"diagnostics\":[");
                boolean first = true;
                for (Diagnostic<? extends JavaFileObject> d : dc.getDiagnostics()) {
                    if (!first) r.append(',');
                    first = false;
                    r.append("{\"kind\":");
                    esc(r, d.getKind() == Diagnostic.Kind.ERROR ? "ERROR" : d.getKind().name());
                    r.append(",\"line\":").append(d.getLineNumber())
                     .append(",\"column\":").append(d.getColumnNumber()).append(",\"code\":");
                    esc(r, String.valueOf(d.getCode()));
                    r.append(",\"message\":");
                    esc(r, oneLine(d.getMessage(Locale.ENGLISH)));
                    r.append('}');
                }
                r.append("]}");
            } catch (Throwable t) {
                r.setLength(0);
                r.append("{\"id\":").append(id).append(",\"ok\":false,\"diagnostics\":[{\"kind\":\"ERROR\",\"line\":-1,\"column\":-1,\"code\":\"server.internal\",\"message\":");
                esc(r, String.valueOf(t));
                r.append("}]}");
            }
            out.print(r.append('\n'));
            out.flush();
        }
    }
}
