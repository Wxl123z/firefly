# Coding Agent 架构学习与研究

**导读**：本项目是一个关于 Coding Agent 架构的学习研究仓库，主要基于互联网上公开的资料以及与DeepSeek讨论整理而成。目前仅实现CLI，包含完整的Agent loop，长上下文管理，工具调用，沙箱校验，Wiki个人知识库构建，重启后的上下文恢复等功能。另外重点研究了模型的工具调用在推理引擎与应用端的传播行为。探索不破坏前缀的渐进式披露工具信息的方法。

这个问题我一开始想错了方向， 查资料以及调试了一个月左右才完成，之后顺手把上下文管理和长期记忆也重做了一遍。这个仓库就是整个研究结果。

代码大概1万1千行 Python，其中逻辑部分约占7K，1 千 8 百行是测试与实验。

[▶ 打开可交互的运行时架构图](https://wxl123z.github.io/firefly/archify/runtime-architecture.html)

也可以下载 [`archify/runtime-architecture.html`](archify/runtime-architecture.html)

---

## 运行

需要 Python 3.12。模型走任何 OpenAI 兼容端点都行，云端或者本地都试过。

```bash
cd agent
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env          # 填 CHAT_API_KEY / CHAT_BASE_URL / EMBED_MODEL
```

```bash
python main.py --profile coding --trace --workspace <PATH>
```

`--workspace`指定工作区，`--trace` 会把每一步的工具调用打出来，第一次跑建议加上，能看清它在干什么。

另外有个端到端演练，会现场建一个带 bug 的 git 仓库让 agent 去修：

```bash
python demo.py
```

```bash
python checks/smoke_test.py --offline    # 不需要 API 的那部分
python checks/smoke_test.py              # 全部
```

---

## 项目结构

```
agent/
├── main.py  demo.py              入口和端到端演练
├── config.py  llm.py             配置、模型和嵌入调用
├── core/
│   ├── loop.py                   对话循环，整个项目的枢纽
│   ├── context.py                四段式 prompt 的组装处
│   ├── profile.py                一个运行时，两套 agent 配置
│   ├── compress.py               滚动摘要压缩
│   ├── plan.py  readset.py       任务清单、已读文件记录
│   └── approval.py  toolsearch.py
├── store/                        SQLite/Postgres + Milvus
├── offline/                      退出时跑的批处理：记忆固化、每日总结
├── personas/                     两套 System Prompt
├── tools/                        注册表、沙箱、工具包
├── wiki/                         派生出来的知识库
├── checks/                       验证装置（占全部代码的四分之一）
└── maintenance/                  一次性运维脚本

archify/
└── runtime-architecture.html                可交互的运行时架构图
```

主链路的两个文件：[`core/loop.py`](agent/core/loop.py)（对话循环）和
[`core/context.py`](agent/core/context.py)（prompt 组装）。

core/profile.py` 里有两套配置，一套是编码 agent，一套是通用助手。
它们共用同一个对话循环、同一套工具基础设施，差别只在工具集、沙箱根、
以及会话隔离。

---

## 一次工具调用，从模型到执行

这段是整个项目最重要的部分，因为大部分问题都出在使用OpenAI SDK与推理引擎交互这条链路的某一环上。

```
输入一句话
   │
   ▼
core/context.py 组装 prompt
   │   前缀：system + 摘要 + 历史        ← 这几段逐字节稳定
   │   尾部：时间 / 水位 / 任务清单 / 当前输入   ← 每轮重算
   ▼
llm.chat_with_tools(messages, tools=[...])
   │
   ▼
模型返回 content
   │   里面可能有文本，也可能有 tool_calls
   ▼
llm.extract_tool_calls() 解析
   │   引擎刮出来的结构化 tool_calls 优先；
   │   没有的话，再从正文里找 <tool_call> 标签
   ▼
core/loop.run_tools() → tools/registry.execute_tool()
   │   校验参数 → 沙箱检查路径 → 审批检查危险操作 → 真的执行
   ▼
结果拼回 messages，回到第一步
```

这条链路上有五层：客户端的 `tools=` 参数、Chat Template 的渲染、模型本身、
输出解析器、还有一道响应闸门。

对应的代码：[`core/context.py`](agent/core/context.py) 负责组装，
[`llm.py`](agent/llm.py) 负责调用与解析，
[`core/loop.py`](agent/core/loop.py) 是那个循环，
[`tools/registry.py`](agent/tools/registry.py) 是执行入口。

---

## 工具注入

起因：早期打算是做出一个search_tools工具，负责在模型调用时以tool_result的身份返回结构化的工具描述信息，但实际测试后发现，模型无法发起正确的工具调用。

我当时的判断是"提示词写得不够清楚"。于是加措辞、加例子、加"必须调用工具"等，但都没用。接着我怀疑是位置问题，把工具定义从系统提示词挪到对话末尾，还是不行。

详细实验请看 [实验-工具尾部注入.md](docs/实验-工具尾部注入.md) 和
[工具调用链路.md](docs/工具调用链路.md) —— 里面有完整的对照组设计、数据和逐层分析。

结论是：判据不在措辞，也不在位置，在于 Chat Template 有没有把工具定义渲染进
`<|im_start|>system` 那个槽位。只要 `tools=` 非空，那个槽位就被占了，模型就无法输出后续得到的工具信息即使模型能看懂甚至可以输出对应的XML标签。

在得到这些数据后，我查了一下模型在推理端的行为，以开源的vllm+Qwen3.5-27B为例，首先在接收到请求后会根据传递的参数，依赖一份jinja模板将完整的上下文渲染成特定的格式，其中，若存在tools参数，有如下逻辑：

```
{%- if tools and tools is iterable and tools is not mapping %}
 {{- '<|im_start|>system\n' }}
 {{- "# Tools\n\nYou have access to the following functions:\n\n" }}
 {%- for tool in tools %}
 {{- "\n" }}{{- tool | tojson }}
 {%- endfor %}
 {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:'
 '\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n'
 'value_1\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n...' }}
 {%- if messages[0].role == 'system' %}
 {%- set content = render_content(messages[0].content, false, true)|trim %}
 {%- if content %}{{- '\n\n' + content }}{%- endif %}
 {%- endif %}
 {{- '<|im_end|>\n' }}
{%- else %}
 ...（普通 system 渲染，没有任何工具段）
{%- endif %}
```

这份模板说明，模型的输出逻辑没有问题，所以最终问题钉在了模型推理引擎，推测是在进行工具调用的输出时采用了引导解码导致模型只能输出已注册tools信息，不过用于验证的实验均通过硅基流动的API进行，且硅基流动的推理引擎是自研且闭源，无法证实，所以也不排除可能是模型在训练阶段所固定下来的格式导致模型无法输出。

基于上述结论，可选的路径有两条：

一：对于云端API，可直接编辑完整的system prompt，手动写入tool调用规则，并在应用端自己写tools调用的解析器解，这样即可完整实现几乎不限数量的tool渐进式披露。

二：也是现在的做法是：常驻工具走 `tools=` 参数，扩展工具放在工具包里，
由模型自己调 `search_tools` 检索，检索到的当场加进 `tools=` 字段。

代价是每加载一次新工具，前缀缓存失效一次，所以加载在会话内是粘性的，不会来回抖。
另外实测模型主动调检索的比例是 83%，剩下那 17% 会直接用核心工具硬干。

---

## 上下文和长期记忆

实验做完之后我把 prompt 整个重排了一遍。现在它是四段：

```
┌────────────────────────────────┐
│ system 提示词                   │  恒定
│ 摘要                            │  很少变
│ 历史消息                        │  追加式增长
╞════════════════════════════════╡  ← 缓存分界
│ 尾部通知（时间/水位/计划/环境）  │  每轮重算
│ 当前输入                        │
├────────────────────────────────┤
│ 本轮工具草稿                     │  回合结束即丢
└────────────────────────────────┘
```

规则只有一条：**会变的东西一律不许进前缀**。时间、上下文水位、可用工具包、
当前任务清单、执行环境——全都塞在尾部。

压缩用的是滚动摘要：把早期消息先固化进结构化记忆，再折叠成摘要，再从上下文里移走。如果只摘要不固化，被丢掉的消息里的具体事实就永久消失了，只剩模型自己写的一段叙述。另外在占用到 80% 的时候有个电路断路器会强制压一次。

长期记忆这块我做了两件事。

一是把对话里的消息增量抽取成结构化记忆，存进 SQLite，同时建向量索引，
所以检索是双源的：既搜原始对话，也搜已经抽象过的记忆。

二是把这些记忆离线整理成一个 markdown 知识库。这个知识库我把它设计成
`f(记忆表, 每日总结表)` 的纯函数 —— 同样的输入重跑一遍，产出的文件**逐字节相同**。
这么做的好处是它坏了不用修，`--rebuild` 从零重建就行。配套写了九项一致性检查
（死链、孤儿页、别名冲突、悬空锚点），检查器自己有反向测试：注入 3 个坏页必须恰好报出 3 个。

---

## 几个踩过的坑

**PowerShell 的退出码。** `$LASTEXITCODE` 只对原生命令有效，cmdlet 失败的时候它是空的，
于是收尾那句 `exit $LASTEXITCODE` 退成了 0。这个 bug 骗过了一次真实任务：
模型往工作区外写文件的命令其实失败了，工具却报"退出码 0（成功）"，
它接着往下做了十步，最后报告完成。修法是在尾部补 `$?` 和执行码归一化。

**陈旧的任务清单。** 计划是存在数据库里的，有一次上一轮遗留了一份"三项全完成"的计划，
模型加载完看到尾部写着"全部完成了，该收尾并汇报"，于是一次工具都没调，
直接宣布任务完成，还编了一整段执行结果。现在已完成的计划不带进下一件事。

**向量索引静默失败。** `except: pass` 把嵌入后端的故障吞了。当时的表现是一切正常 ——
检索照样能用，只是在那 16 条里搜。直到我偶然去看 Milvus 的行数，
才发现七千多条消息里只索引了 16 条。现在加了失败计数和节流上报，
另外有个幂等的重建脚本。

**压缩之后的写保护。** 这个是最晚发现的。`readset` 记录"本会话读过哪些文件"，
`edit_file` 靠它挡住"凭想象编辑"。但压缩会把消息移出上下文，
于是会出现这种情况：模型读过 `calc.py`，压缩把它移走了，
模型想改它，守卫查记录说"读过，放行"—— 而它已经看不见文件内容了。
守卫想保证的是"模型此刻看得见"，实际检查的是"这个会话里读过"，
压缩把这两件事分开了。修法是压缩成功之后把读过的记录一并作废。

---

## 一些我知道的局限

**`run_shell` 绕得过沙箱。** 工具层的路径检查只约束参数，
命令字符串里 `cd ..` 就能出去。这个在工具层堵不住，正确的位置是审批，
我暂时把它当成已知限制记着，没做限制。

**目前所有实测都是单机、单会话、几千条消息的规模。** 没做过多用户、大仓库、
跨月的验证。

**语义检索和关键词检索的质量无法验证。**同样是由于目前仅 个人用户且没有累计对话历史。

**空闲时间的利用还没做。** 现在所有离线整理都压在退出时一次性做完，
理想情况应该是模型在没人说话的时候自己安排，我做了初步设计但还没动手。
