Download Link: https://assignmentchef.com/product/solved-ceng462-artificial-intelligence-homework-iv-a-new-hope
<br>
<h1>1             Objectives</h1>

This assignment aims to assist you to expand your knowledge on Reinforcement Learning in case of Value Iteration and Q-Learning methods.

<h1>2             Problem Definition</h1>

<em>“A long time ago, in a galaxy far, far away…</em>

<em>It is a period of civil war. Rebel spaceships, striking from a hidden base, have won their first victory against the evil Galactic Empire. During the battle, Rebel spies managed to steal secret plans to the Empire’s ultimate weapon, the DEATH STAR, an armored space station with enough power to destroy an entire planet. Pursued by the Empire’s sinister agents, Princess Leia races home aboard her starship, custodian of the stolen plans that can save her people and restore freedom to the galaxy…”</em>

As you may already know, this is the opening text from the fourth episode of the famous <em>Star Wars </em>movie. Without giving further spoiler, <em>Princess Leia </em>needs <em>R2D2</em>’s help to get the message to <em>Obi-Wan Kenobi </em>and as being an agent, <em>R2D2 </em>is a close friend of us, so we would like to help it in its divine struggle.

Figure 1: <em>Princess Leia </em>and our friend, <em>R2D2</em>

In this assignment, you will help <em>R2D2 </em>by using the <em>force </em>of Reinforcement Learning. You will be given a problem domain like below and apply <strong>Value Iteration </strong>or <strong>Q-Learning </strong>on this domain according to given configurations. For such a problem, our agent, <em>R2D2</em>, should reach the goal state, <em>Obi-Wan Kenobi</em>, avoiding possible pitfalls as <em>stormtroopers </em>and from any state in which it is placed.

Figure 2: A possible scene from an example problem domain where <em>R2D2</em>, a <em>stormtrooper </em>and <em>Obi-Wan Kenobi </em>are in (1,1), (5,3) and (5,5) coordinates respectively.

<h1>3             Specifications</h1>

<ul>

 <li>You will write a python script which will read the problem configuration from a file and output the policy constructed by the requested method to a file.</li>

 <li>In <strong>Value Iteration </strong>you will check the maximum difference between old and updated V values within a given thresold for the termination of process as in the lecture materials.</li>

 <li>For <strong>Q-Learning</strong>;

  <ul>

   <li>The learning process will be episodic. An episode will end only when the agent reaches the goal state.</li>

   <li>In each episode, the agent will start from a random state excluding the goal state and the states which are pitfalls and obstacles.</li>

   <li>You will use -greedy approach for action selection mechanism. In other words, within a given probability of , the agent will choose a random action rather than the following its current policy.</li>

  </ul></li>

 <li>For the given problem,

  <ul>

   <li>The coordinate of (1,1) will be the left-lower corner of the environment. X-coordinate will increase while going east and Y-coordinate will increase while going north.</li>

   <li>The domain size is not fixed, the length and width of the environment will be given as inputs.</li>

   <li>The environment will be static regarding the possible obstacles, pitfalls and goal state. That is, their positions will not change during learning.</li>

   <li>The agent can move in four directions in the environment: north, east, south and west.</li>

   <li>If the agent hits a wall around the domain (tries to get out of domain boundaries) or an obstacle, it should stay in the current cell.</li>

   <li>The environment will be deterministic, so any movement of the agent should result in only one way, considering the conditions above.</li>

   <li>The agent may receive four different rewards from the environment: (i) <strong>regular reward </strong>for default step of the agent which don’t cause any hit or don’t make the agent reach the goal, (ii) <strong>obstacle reward </strong>for hitting a wall or an obstacle, (iii) <strong>pitfall reward </strong>for getting damaged by a pitfall, (iv) <strong>goal reward </strong>for reaching the goal state. They can be <strong>any real number </strong>as long as they are consistent with their meanings.</li>

  </ul></li>

</ul>

<h1>4             I/O Structure</h1>

Your program will read the input from a file whose path will be given as the first command-line argument. The first line of the input file will be the method as “V” for Value Iteration or “Q” for Q-Learning. The rest of file will differ according to requested method. Hence, the structures of these two different input types are given separately.

The structure of input which requires Value Iteration:

<table width="673">

 <tbody>

  <tr>

   <td width="226">V<em>&lt;</em>theta<em>&gt;</em></td>

   <td width="447"># termination condition factor</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>gamma<em>&gt;</em></td>

   <td width="447"># discount factor</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>M<em>&gt; &lt;</em>N<em>&gt;</em></td>

   <td width="447"># dimensions (M :y−dimension , N :x−dimension)</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>k<em>&gt;</em></td>

   <td width="447"># number of obstacles</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>o_1_x<em>&gt; &lt;</em>o_1_y<em>&gt;</em></td>

   <td width="447"># coordinates of obstacle 1</td>

  </tr>

  <tr>

   <td width="226">. . .<em>&lt;</em>o_k_x<em>&gt; &lt;</em>o_k_y<em>&gt;</em></td>

   <td width="447"># coordinates of obstacle k</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>l<em>&gt;</em></td>

   <td width="447"># number of pitfalls</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>p_1_x<em>&gt; &lt;</em>p_1_y<em>&gt;</em></td>

   <td width="447"># coordinates of pitfall 1</td>

  </tr>

  <tr>

   <td width="226">. . .<em>&lt;</em>p_l_x<em>&gt; &lt;</em>p_l_y<em>&gt;</em></td>

   <td width="447"># coordinates of pitfall l</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>g_x<em>&gt; &lt;</em>g_y<em>&gt;</em></td>

   <td width="447"># coordinates of goal state</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>r_d<em>&gt; &lt;</em>r_o<em>&gt; &lt;</em>r_p<em>&gt; &lt;</em>r_g<em>&gt;</em></td>

   <td width="447"># rewards of regular step , hitting an obstacle/wall , getting damaged by pitfall , reaching the goal respectively</td>

  </tr>

 </tbody>

</table>

The structure of input which requires Q-Learning:

<table width="673">

 <tbody>

  <tr>

   <td width="226">Q<em>&lt;</em>number of episode<em>&gt;</em><em>&lt;</em>alpha<em>&gt;</em></td>

   <td width="447"># learning rate</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>gamma<em>&gt;</em></td>

   <td width="447"># discount factor</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>epsilon<em>&gt;</em></td>

   <td width="447"># paramater for −greedy approach</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>M<em>&gt; &lt;</em>N<em>&gt;</em></td>

   <td width="447"># dimensions (M :y−dimension , N :x−dimension)</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>k<em>&gt;</em></td>

   <td width="447"># number of obstacles</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>o_1_x<em>&gt; &lt;</em>o_1_y<em>&gt;</em></td>

   <td width="447"># coordinates of obstacle 1</td>

  </tr>

  <tr>

   <td width="226">. . .<em>&lt;</em>o_k_x<em>&gt; &lt;</em>o_k_y<em>&gt;</em></td>

   <td width="447"># coordinates of obstacle k</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>l<em>&gt;</em></td>

   <td width="447"># number of pitfalls</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>p_1_x<em>&gt; &lt;</em>p_1_y<em>&gt;</em></td>

   <td width="447"># coordinates of pitfall 1</td>

  </tr>

  <tr>

   <td width="226">. . .<em>&lt;</em>p_l_x<em>&gt; &lt;</em>p_l_y<em>&gt;</em></td>

   <td width="447"># coordinates of pitfall l</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>g_x<em>&gt; &lt;</em>g_y<em>&gt;</em></td>

   <td width="447"># coordinates of goal state</td>

  </tr>

  <tr>

   <td width="226"><em>&lt;</em>r_d<em>&gt; &lt;</em>r_o<em>&gt; &lt;</em>r_p<em>&gt; &lt;</em>r_g<em>&gt;</em></td>

   <td width="447"># rewards of regular step , hitting an obstacle/wall , getting damaged by pitfall , reaching the goal respectively</td>

  </tr>

 </tbody>

</table>

As output, your program will create a file, whose path will be given as the second command-line argument. This file should include the policy represented by as following structure:

<table width="673">

 <tbody>

  <tr>

   <td width="210"><em>&lt;</em>s_1_x<em>&gt; &lt;</em>s_1_y<em>&gt; &lt;</em>a_1<em>&gt;</em></td>

   <td width="463"># coordinates of state 1 and chosen action in this state (as 0: ’N’ , 1: ’E’ , 2: ’S ’ , 3: ’W’ )</td>

  </tr>

  <tr>

   <td width="210"><em>&lt;</em>s_2_x<em>&gt; &lt;</em>s_2_y<em>&gt; &lt;</em>a_2<em>&gt;</em></td>

   <td width="463"># coordinates of state 2 and chosen action in this state</td>

  </tr>

  <tr>

   <td width="210">. . .<em>&lt;</em>s_i_x<em>&gt; &lt;</em>si_y<em>&gt; &lt;</em>a_i<em>&gt;</em></td>

   <td width="463"># coordinates of state i (i = MxN) and chosen action in this state</td>

  </tr>

 </tbody>

</table>

<strong>Important Notes:</strong>

<ul>

 <li>In all files, space character is used as delimiter of multiple entries in a line.</li>

 <li>An example file for each structure can be found in homework file on OdtuClass. Make sure your code is compatible with them.</li>

</ul>

<h1>5             Regulations</h1>

<ol>

 <li><strong>Programming Language: </strong>You must code your program in Python <strong>7</strong>. Your submission will be tested in inek machines. So make sure that it can be executed on them as below.</li>

</ol>

python e1234567_hw4 . py input . txt output . txt

<ol start="2">

 <li><strong>Implementation: </strong>You are allowed to use <strong>sys </strong>and <strong>random </strong>modules from standard python library. You cannot import any other modules.</li>

 <li><strong>Late Submission: </strong>No late submission is allowed. No correction can be done in your codes after deadline.</li>

 <li><strong>Cheating: We have zero tolerance policy for cheating</strong>. People involved in cheating (any kind of code sharing and codes taken from internet included) will be punished according to the university regulations.</li>

 <li><strong>Discussion: </strong>You must follow the OdtuClass for discussions and possible updates on a daily basis.</li>

 <li><strong>Evaluation: </strong>Your program will be evaluated automatically by simulating the policy you recorded in the output file, so make sure to obey the specifications. As long as your policy reaches an optimal solution for the given testcase, for reasonable amount of trials starting from a random regular state, your code will pass that case. A reasonable timeout will be applied according to the complexity of test cases in order to avoid infinite loops due to an erroneous code.</li>

</ol>