# interviewbit-imp-question
*****************************************************************************************
    bool compare(Interval A,Interval B) 
    {
        return A.start < B.start;
    }
    vector<Interval> Solution::merge(vector<Interval> &A) {
    int n=A.size(); 
    sort(A.begin(),A.end(),compare); 
    //for(int i=0;i<A.size();i++) 
        //cout<<A.start<<" ";
    vector<Interval> ans;
    Interval b=Interval(0,0); 
    for(int i=0;i<A.size();i++) 
    {
        if(i==0) 
        {
            b.start=A[i].start;
            b.end=A[i].end;
        } 
        else 
        {
            if(A[i].start <= b.end) 
                b.end=max(b.end,A[i].end);  // change here for overlapping with same value 
            else 
            {
                ans.push_back(b);
                b=Interval(A[i].start,A[i].end);
            }  
        }
    } 
    ans.push_back(b);
    return ans;
    }
*****************************************************************************************
      vector<int> Solution::nextPermutation(vector<int> &A) {
          next_permutation(A.begin(),A.end());
          return A;
      }

*****************************************************************************************
      class Solution {
      public:
          vector<int> repeatedNumber(const vector<int> &V) {
             long long sum = 0;
             long long squareSum = 0;
             long long temp;
             for (int i = 0; i < V.size(); i++) {
                 temp = V[i];
                 sum += temp;
                 sum -= (i + 1);
                 squareSum += (temp * temp);
                 squareSum -= ((long long)(i + 1) * (long long)(i + 1));
             }
             // sum = A - B
             // squareSum = A^2 - B^2 = (A - B)(A + B)
             // squareSum / sum = A + B
             squareSum /= sum;

             // Now we have A + B and A - B. Lets figure out A and B now. 
             int A = (int) ((sum + squareSum) / 2);
             int B = squareSum - A;

             vector<int> ret;
             ret.push_back(A);
             ret.push_back(B);
             return ret;
          }
      };

*****************************************************************************************
      int Solution::repeatedNumber(const vector<int> &A) {
          const int k = 3;
          int n = A.size();
          vector<int> cnt(k, 0), candidate(k, -1);
          for (int i = 0; i < n; i++) {
              bool updated = false;
              for (int j = 0; j < k; j++) {
                  if (candidate[j] == A[i]) {
                      cnt[j]++, updated = true;
                      break;
                  }
              }
              if (updated) continue;
              for (int j = 0; j < k; j++) {
                  if (cnt[j] == 0) {
                      cnt[j]++, updated = true;
                      candidate[j] = A[i];
                      break;
                  }
              }
              if (updated) continue;
              for (int j = 0; j < k; j++)
                  cnt[j]--;
          }
          cnt.assign(k, 0);
          for (int i = 0; i < n; i++) {
              for (int j = 0; j < k; j++) {
                  if (candidate[j] == A[i]) {
                      cnt[j]++;
                      break;
                  }
              }
          }
          for (int j = 0; j < k; j++)
              if (cnt[j] > n / k) return candidate[j];
          return -1;
      }
*****************************************************************************************
       vector<Interval> Solution::insert(vector<Interval> &intervals, Interval newInterval) {
         vector<Interval>v;
          int n=intervals.size();

          for(i=0;i<(n-1);i++){
              if(intervals[i].end>=intervals[i+1].start){

              }
          }
      }
******************************************************************************************
      vector<int> Solution::flip(string A) {
          int meh=0,m=INT_MIN,l=1,r=1,l1=1,count=0;
          for(int i=0;i<A.size();i++){
              if(A[i]=='0'){
                  meh++;
                  count++;
                  if(m<meh){
                      m=meh;
                      r=i+1;
                      l=l1;
                  }
              }else{
                  meh--;
                  if(meh<0){
                      meh=0;
                      l1=i+2;
                  }
              }
          }
          if(count==0)
              return {};
          return {l,r};
      }

****************************************************************************************
      void helper(vector<int>&A,int i,int j,vector<vector<int>>&out,int n){
          if(i==n){
              out.push_back(A);
              return;
          }
          for(j=i;j<n;j++){
              swap(A[i],A[j]);
              helper(A,i+1,j+1,out,n);
              swap(A[i],A[j]);
          }
          return;
      }

      vector<vector<int> > Solution::permute(vector<int> &A) {
          vector<vector<int>>out;
          helper(A,0,0,out,A.size());
          return out;
      }

*****************************************************************************************
            vector<vector<int> > Solution::subsetsWithDup(vector<int> &A) {
                set<vector<int>>v;
                sort(A.begin(),A.end());
                vector<int>subset;
                ans(v,subset,A,0);
                vector<vector<int>>a(v.begin(),v.end());
                return a;
            }
****************************************************************************************
            solve(vector<int> &A,int n, vector<int> &temp,set<vector<int>> &ans,int index=0){
                if(index == n){
                    ans.insert(temp);
                    return ;
                }
                if(index >n)
                    return ;
                for(int j=)
            }

            vector<vector<int> > Solution::permute(vector<int> &A) {
                int n=A.size();
                set<vector<int>>ans;
                solve(A,n,temp,ans);
                vector<vector<int>>res;
                for(auto x: ans)
                    res.push_back(x);
                return res;
            }

***************************************************************************************
            void generate(int A,vector<string> &vec, string s="", int unbalance=0){
                if(s.length() == 2*A and not unbalance){
                    vec.push_back(s);
                    return;
                }
                if(s.length() >2*A) return;
                if(unbalance < 0 or unbalance > A)return;
                generate(A,vec,s + "(", unbalance + 1);
                generate(A,vec,s + ")", unbalance - 1);
            }

            vector<string> Solution::generateParenthesis(int A) {
                vector<string>vec;
                generate(A,vec);
                return vec;
            }
****************************************************************************************
            void solve(string &A,vector<string> &vec,string s="",int index=0){
                if(index==A.length()){
                    vec.push_back(s);
                }
                if(s[i]=="2"){

                }
            }
            vector<string> Solution::letterCombinations(string A) {
                vector<string>vec;
                solve(A,vec);
                return vec;

            }

*****************************************************************************************
            void generate(vector<int> &vec, int A, int curVal, bool turn = true) {
                if (not A) {
                    vec.push_back(curVal);
                    return;
                }
                if (turn) {
                    generate(vec, A-1, (curVal<<1)+0, turn);
                    generate(vec, A-1, (curVal<<1)+1, !turn);
                } else {
                    generate(vec, A-1, (curVal<<1)+1, !turn);
                    generate(vec, A-1, (curVal<<1)+0, turn);
                }
            }

            vector<int> Solution::grayCode(int A) {
                vector<int> vec;
                generate(vec, A, 0);
                return vec;
            }

***************************************************************************************
            void backtrack(vector &A,int index,vector<vector> &ans,vector &temp){
                ans.push_back(temp);
                for(int i=index;i<A.size();i++){
                    temp.push_back(A[i]);
                    backtrack(A,i+1,ans,temp);
                    temp.pop_back();
                }
            }
            vector<vector<int> > Solution::subsets(vector<int> &A) {
                vector<vector> ans;
                vector temp;
                sort(A.begin(),A.end());
                backtrack(A,0,ans,temp);
                return ans;
            }

***************************************************************************************
            bool isvalid(vector<vector<char>> &b, int r, int c, char k) {

                for(int i = 0; i<9; ++i){
                    if(b[r][i] == k) return false;
                }

                for(int i = 0; i<9; ++i){
                    if(b[i][c] == k) return false;
                }

                int blockx = r/3;
                int blocky = c/3;
                for(int i = 3*blockx; i< 3*blockx+3; ++i){
                    for(int j = 3*blocky; j<3*blocky+3; ++j){
                        if(b[i][j] == k) return false;
                    }
                }
                return true;
            }

            bool solver(vector<vector<char>> &b){
                for(int i = 0; i < 9; ++i){
                    for(int j = 0; j<9; ++j){
                        if(b[i][j] == '.'){
                            for(int k = 1; k < 10; ++k){
                               if(isvalid(b, i, j, (char)('0'+k))){
                                    b[i][j] = (char)('0' + k);

                                    if(solver(b)) return true;
                                    b[i][j] = '.';
                               }
                            }
                            return false;
                        }
                    }
                }

                return true;
            }
            void Solution::solveSudoku(vector<vector<char> > &A) {
                solver(A);
            }

***************************************************************************************
            bool isValid(vector<string> &board, string &s, int row, int col, int n){
                if(s[col]=='Q') return false; // for current row

                int j=col, k=1;
                for(int i=row-1; i>=0; i--){
                    if(j+k<n && board[i][j+k]=='Q') // upper right diagonal
                        return false;
                    if(j-k>=0 && board[i][j-k]=='Q') // upper left diagonal
                        return false;
                    k++;
                }
                return true;
            }

            void backtracker(vector<vector<string>> &soln, vector<string> &board, int row, int n, string &s){
                if(row==n){
                    soln.push_back(board);
                    return;
                }
                for(int col=0; col<n; col++){
                    if(isValid(board, s, row, col, n)){
                        board[row][col] = 'Q'; s[col] = 'Q';
                        backtracker(soln, board, row+1, n, s);
                        board[row][col] = '.'; s[col] = '.';
                    }
                }
            }

            vector<vector<string> > Solution::solveNQueens(int A) {
                vector<vector<string>> soln;
                vector<string> board(A, string(A,'.'));
                string s(A, '.');
                backtracker(soln, board, 0, A, s);
                return soln;
            }

**************************************************************************************
            int Solution::searchInsert(vector<int> &A, int B) {
                int n = A.size();
                if(B>A[n-1])
                    return n;
                int start=0;
                int end = n-1;
                int ans;
                while(start<=end)
                {
                    int mid = start + (end-start)/2;
                    if(A[mid]==B)
                        return mid;
                    if(A[mid]>B)
                    {
                        ans = mid;
                        end = mid-1;
                    }
                    else
                        start = mid+1;
                }
                return ans;

            }
***************************************************************************************
            void helper( vector<int> A,int B, int start , int end , int &mn)
            {
                while(start<=end)
                {
                    int mid = start + (end - start)/2;
                    if(A[mid]==B)
                        mn = min(mn,mid);
                    if(A[mid] < B)
                        start = mid+1;
                    else 
                        end = mid-1;
                }
            }
            void helper1( vector<int> A,int B, int start , int end , int &mx)
            {
                while(start<=end)
                {
                    int mid = start + (end - start)/2;
                    if(A[mid]==B)
                        mx = max(mx,mid);
                    if(A[mid] <= B)
                        start = mid+1;
                    else 
                        end = mid-1;
                }
            }

            vector<int> Solution::searchRange(const vector<int> &A, int B) {
                int n = A.size();
                int mn =INT_MAX,mx=INT_MIN;
                helper(A,B,0,n-1,mn);
                helper1(A,B,0,n-1,mx);
                if(mn == INT_MAX)
                    mn = -1;
                if(mx == INT_MIN)
                    mx = -1;
                return vector<int> {mn,mx};
            }
****************************************************************************************
            double Solution::findMedianSortedArrays(const vector<int> &A, const vector<int> &B) {
                    int arr[A.size()],brr[B.size()];
                    for(int i=0;i<A.size();i++)arr[i]=A[i];
                    for(int i=0;i<B.size();i++)brr[i]=B[i];

                    int l1=A.size();
                    int l2=B.size();
                    if(l1>l2){
                        int end=l1-1;
                        int start=0;
                        for(int i=0;i<l2;i++){
                            if(brr[end]<arr[start]){
                                swap(brr[end],arr[start]);
                                end--;
                                start++;
                            }else{
                              break;  
                            }
                        }
                        sort(arr,arr+l1);
                        return arr[(l1+l2)/2];
                    }else{

                    }

            }

***************************************************************************************
            int Solution::singleNumber(const vector<int> &A) {
                int ans=0,len=A.size(),even,odd,p=1;

                for(int i=0;i<32;i++){
                    even=0;
                    odd=0;
                    for(int j=0;j<len;j++){
                        if(p&A[j]==0){
                            even++;
                        }else odd++;
                    }
                    if(odd%3!=0)
                        ans+=p;
                    p*=2;
                }
                return ans;

            }
**************************************************************************************
            int solve(TreeNode* A, int &res){
                 if(A==NULL)return 0;
                int l=solve(A->left,res);
                int r=solve(A->right,res);

                int temp=max(max(l,r)+A->val,A->val);

                int ans=max(temp,l+r+A->val);
                res = max(res,ans);

                return temp;

             }


            int Solution::maxPathSum(TreeNode* A) {
                int res=INT_MIN;
                solve(A,res);
                return res;
            }
***************************************************************************************
            int Solution::solve(vector<int> &A, int B) {
                int n=A.size();
                int pre[n+4]={0},suf[n+4]={0};

                int i,j,k;

                for(i=1;i<=n;i++){
                    pre[i]=min(pre[i-1],A[i-1]);
                }
                for(i=(n-1);i>=0;i--){
                    suf[i+1]=max(suf[i+2],A[i]);
                }
                int diff[n+4]={0};
                for(i=1;i<=n;i++){
                    diff[i]=(suf[i]-pre[i]);
                }
                sort(diff+1,diff+n+1);
                int sum=0;
                for(i=n;i>=(n-B+1);i--){
                    sum+=diff[i];
                }

                return sum;

            }
***************************************************************************************
            vector<vector<int> > Solution::solve(int A, vector<vector<int> > &B) {
                if(A==0)    
                    return B;
                int r=B.size();
                int c=B[0].size();

                vector<vector<int>>C=B;

                for(int m=1;m<=A;m++){
                    for(int i=0;i<r;i++){
                        for(int j=0;j<c;j++){
                            int neigh_x[] = {0,1,0,-1};
                            int neigh_y[] = {1,0,-1,0}

                            int max_neigh = B[i][j];
                            for(int t=0;t<4;t++){
                                int neigh_pos_x=i+neigh_x[t];
                                int neigh_pos_y=j+neigh_y[t];
                                if(neigh_pos_x>=0 and neigh_pos_x<r and neigh_pos_y>=0 and neigh_pos_y<c){
                                    max_neigh = max(max_neigh, B[neigh_pos_x][neigh_pos_y]);
                                }
                            }

                            C[i][j]=max_neigh;

                         }
                    }
                    B=C;
                }

                return C;

            }

***************************************************************************************
            int Solution::maxProfit(const vector<int> &A) {
                if(A.size()==0 or A.size()==1)return 0;
                int fb=INT_MIN, sb=INT_MIN, fs=INT_MIN, ss=INT_MIN;
                for(int i=0;i<A.size();i++){
                    fb=max(fb,-A[i]);
                    fs=max(fs,fb+A[i]);
                    sb=max(sb,fs-A[i]);
                    ss=max(ss,sb+A[i]);

                }
                return ss;
            }

****************************************************************************************
            int Solution::isMatch(const string A, const string B) {
                int n=A.length();
                int m=B.length();
                bool dp[n+2][m+2];
                memset(dp,false,sizeof(dp));

                dp[0][0]=true;
                for(int i=1;i<=n;i++)
                    dp[i][0]=false;
                for(int i=1;i<=m;i++){
                    dp[0][i]=false;
                    if(i>=2 and B[i-1]=='*')
                        dp[0][i]=dp[0][i-2];
                }
                for(int i=1;i<=n;i++){
                    for(int j=1;j<=m;j++){
                        if(A[i-1]==B[j-1] || B[j-1]=='.')
                            dp[i][j]=dp[i-1][j-1];
                        else if(B[j-1]=='*'){
                            if(j>=2 and (A[i-1]==B[j-2] || B[j-2]=='.'))
                                dp[i][j]=dp[i-1][j]|dp[i][j-2];
                            else if(j>=2)
                                dp[i][j]=dp[i][j-2];
                        }
                    }
                }
                return dp[n][m];

            }

**************************************************************************************
            int compute(vector<int> &A, int l, int r, bool turn, vector<vector<int> > &dp) {
                if(l>r)                     return 0;
                if(l==r)                    return turn ? A[l] : 0;
                if(dp[l][r] != -1)          return dp[l][r];
                int left = compute(A, l+1, r, !turn, dp);
                int right = compute(A, l, r-1, !turn, dp);
                if(turn)
                    dp[l][r] = max(left+A[l], right+A[r]);
                else
                    dp[l][r] = min(left, right);
                return dp[l][r];
            }
            int Solution::maxcoin(vector<int> &A) {
                int len = A.size();
                vector<vector<int> > dp(len, vector<int>(len, -1));
                dp[0][len-1] = compute(A, 0, len-1, true, dp);
                return dp[0][len-1];
            }
****************************************************************************************
            int dp[701][701];
            int func(int i,int j,string A,string B){
                if(i>=A.size() || j>=B.size()){
                    if(j==B.size())return 1;
                    return 0;

                }
                if(dp[i][j]!=-1)return dp[i][j];
                dp[i][j]=0;
                if(A[i]==B[j])
                    dp[i][j]+=func(i+1,j+1,A,B);
                dp[i][j]+=func(i+1,j,A,B);
                return dp[i][j];

            }

            int Solution::numDistinct(string A, string B) {
                memset(dp,-1,sizeof dp);
                return func(0,0,A,B);
            }

****************************************************************************************
            void wordbreak(int index, string &s, string temp,vector<string> &res, unordered_map<string,int> &mp){
                if(index == s.size()){
                    temp.pop_back();
                    res.push_back(temp);
                    return;
                }
                string sub="";
                for(int i=index;i<s.size();i++){
                    sub+=s[i];
                    if(mp.find(sub)!=mp.end()){

                    }
                }
            }

            vector<string> Solution::wordBreak(string A, vector<string> &B) {
                vector<string>res;
                unordered_map<string,int>mp;
                for(int i=0;i<B.size();i++)
                    mp[B[i]]++;
                wordbreak(0,A,"",res,mp);
                return res;
            }

**************************************************************************************
            bool dfs(string A, int ind, vector<int> &dp,unordered_map<string, int> &dict,int ms){
                if(A.size() == ind)return true;
                if(dp[ind]!=-1)return (bool)dp[ind];
                bool flag=false;
                string s="";
                for(int i=ind;i<A.size();i++){
                    s+=A[i];
                    if(s.size()>ms)break;
                    if(dict[s])flag|=dfs(A,i+1,dp,dict,ms);
                }
                return dp[ind]=flag;
            }

            int Solution::wordBreak(string A, vector<string> &B) {
                unordered_map<string,int>dict;
                vector<int>dp(A.size()+1,-1);
                int ms=0;
                for(auto x: B)ms=max(ms,(int)x.size()),dict[x]++;
                return (int)dfs(A,0,dp,dict,ms);
            }

***************************************************************************************
            int Solution::solve(const vector<int> &A, const vector<int> &B, const vector<int> &C) {
                int t = *max_element(A.begin(), A.end());
                int dp[t+1];
                memset(dp,-1,sizeof(dp));
                dp[0]=0;
                int k=0;
                for(auto i:B){
                    for(int j=i;j<=t;j++){
                        if(dp[j-i]!=-1)
                            dp[j]=dp[j]==-1 ? dp[j-i]+C[k]:min(dp[j],dp[j-i]+C[k]);
                    }
                    k++;
                }
                int ans=0;
                for(auto i:A)ans+=dp[i];
                return ans;

            }

****************************************************************************************
            int solve(vector<vector<int>> &A,int s,int e, int n,int m,vector<vector<int>> &dp){
                if((s+1)==n and (e+1)==m)   
                    return A[s][e];
                if(dp[s][e]!=-1)
                    return dp[s][e];
                if(s+1==n)
                    return dp[s][e]=min(A[s][e],A[s][e]+solve(A,s,e+1,n,m,dp));
                else if(e+1==m)
                    return dp[s][e]=min(A[s][e],A[s][e]+solve(A,s+1,e,n,m,dp));
                else
                    return dp[s][e]=max(min(A[s][e],A[s][e]+solve(A,s+1,e,n,m,dp)),
                                        min(A[s][e],A[s][e]+solve(A,s,e+1,n,m,dp)));
            }

            int Solution::calculateMinimumHP(vector<vector<int> > &A) {
                int n,m;
                n=A.size();
                m=A[0].size();
                vector<vector<int>>dp(n,vector(m,-1));
                int ans=solve(A,0,0,n,m,dp);
                if(ans>0)
                    return 1;
                return abs(ans-1);
            }

**************************************************************************************
            int MOD=1000000007;

            int func(vector<int>& arr,int sum,int k,vector<vector<int>>& t){
                if(sum==0 and k==0)
                    return 1;
                if(sum<0)
                    return 0;
                if(k==0)
                    return 0;
                if(t[sum][k]!=-1)
                    return t[sum][k];
                int ans=0;
                for(int i=0;i<arr.size();i++){
                    if(arr[i]<=sum)
                        ans=((ans%MOD)+(func(arr,sum-arr[i],k-1,t)%MOD))%MOD;
                }
                return t[sum][k]=ans%MOD;
            }

            int Solution::solve(int A, int B) {
                vector<vector<int>>t(1000, vector<int>(1000,-1));
                vector<int> arr = {1,2,3,4,5,6,7,8,9,0};
                int ans=0;
                for(int i=0;i<9;i++){
                    ans=((ans%MOD) + (func(arr,B-arr[i],A-1,t)%MOD))%MOD;
                }
                return ans%MOD;

            }
****************************************************************************************
            int cutRod(int n, vector<int> &B){
                int val[n+1];
                val[0]=0;
                int i,j;
                for(i=1;i<=n;i++){
                    int max_val=INT_MIN;
                    for(j=0;j<i;j++){
                        max_val=max(max_val,B[j]+val[i-j-1]);
                    }
                    val[i]=max_val;
                }
                return val[n];
            }

            vector<int> Solution::rodCut(int A, vector<int> &B) {
                return cutRod(A, B);
            }

*****************************************************************************************
            unordered_map<UndirectedGraphNode*, UndirectedGraphNode*>mp;

            UndirectedGraphNode dfs(UndirectedGraphNode Node){
                if(node == NULL)
                    return NULL:
                if(mp.find(node) == mp.end()){
                    mp[node] = new UndirectedGraphNode(node->label);
                    for(auto x:node->neighbors){
                        mp[node]->neighbors.push_back(dfs(x));
                    }
                }
                return mp[node];
            }

            UndirectedGraphNode *Solution::cloneGraph(UndirectedGraphNode *node) {
                mp.clear();
                return dfs(node);
            }

***************************************************************************************
            int find1(int a1[],int x){
                if(a1[x]== -1)return x;
                return a1[x]=find(a1,a1[x]);
            }

            int Solution::solve(int A, vector<int> &B, vector<int> &C) {
                if(B.size()>=A)
                    return 0;
                else
                    return 1;
                if(!A)
                    return 0;
                vector<vector<int>>graph(A+1,vector<int>());
                int i=0,size=min(B.size(),C.size());
                for(i=0;i<size;i++)
                    graph[B[i]].push_back(C[i]);
                vector<bool>visited(A+1,false);
                vector<bool>revStack(A+1,false);

                for(i=1;i<size;i++)
                    if(!visited[i] and isCyclic(graph,i,visited, recStack))
                        return 0;
                return 1;
            }

***************************************************************************************
            int Solution::solve(vector<int> &A) {
                vector<int>hgt(A.size(),0);
                int ans=0,maxx=0;
                for(int i=A.size()-1;i>0;i--){
                    ans=max(ans,hgt[A[i]]+hgt[i]+1);
                    hgt[A[i]]=max(hgt[i]+1,hgt[A[i]]);
                }
                return ans;
            }

****************************************************************************************
            bool isInside(int circle_x,int circle_y,int rad,int x,int y){
                return (pow((x-circle_x),2) + pow((y-circle_y),2))<=pow(rad,2);
            }

            string Solution::solve(int A, int B, int C, int D, vector<int> &E, vector<int> &F) {
                vector<vector<int>>visited(A+1,vector<int>(B+1,0));
                queue<pair<int,int>>q;
                for(int i=0;i<A;i++){
                    for(int j=0;j<B;j++){
                        for(int k=0;k<E.size();k++){
                            if(isInside(E[k],F[k],D,i,j))
                                visited[i][j]=-1;
                        }
                    }
                }
                if(visited[0][0]==-1)return "NO";

                q.push({0,0});
                visited[0][0]=1;
                while(!q.empty()){
                    auto curr=q.front();q.pop();
                    int x = curr.first;
                    int y= curr.second;

                }

            }

**************************************************************************************
            vector<int> Solution::solve(vector<int> &A, vector<int> &B, int C) {
                vector<int>res;
                priority_queue<tuple<int, int, int>>q;
                sort(A.rbegin(), A.rend()); 
                sort(B.rbegin(), B.rend());
                for(int i=0; i<C; i++) {
                    q.push({A[i]+B[0], i, 0});
                }
                while(res.size()<C) {
                    auto [sum, i, j] = q.top(); q.pop();
                    res.push_back(sum);
                    q.push({A[i]+B[j+1], i, j+1});
                }
                return res;
            }

***************************************************************************************
            int isRectangle(const vector<vector<int> >& matrix) {
                int rows = matrix.size();
                if (rows == 0)
                    return 0;

                int columns = matrix[0].size();
                unordered_map<int, unordered_set<int> > table;

                for (int i = 0; i < rows; ++i) {
                 for (int j = 0; j < columns - 1; ++j) {
                    for (int k = j + 1; k < columns; ++k) {
                      if (matrix[i][j] == 1 && matrix[i][k] == 1) {
                        if (table.find(j) != table.end() && table[j].find(k) != table[j].end())
                                    return 1;
                        if (table.find(k) != table.end() && table[k].find(j) != table[k].end())
                                    return 1;
                        table[j].insert(k);
                        table[k].insert(j);
                      }
                    }
                  }
                }
                return 0;
            }


            int Solution::solve(const vector<vector<int> > &A) {
                return isRectangle(A);
            }
****************************************************************************************
            #include<list>
            list<int> q;
            unordered_map<int,pair<int,list<int>::iterator>> m;
            int cap;
            LRUCache::LRUCache(int capacity) {
                 cap = capacity;
                 q.clear();
                 m.clear();
            }

            int LRUCache::get(int key) {
                if(m.find(key)!= m.end()){
                    q.erase(m[key].second);
                    q.push_front(key);
                    m[key].second = q.begin();
                    return m[key].first;
                }
                else
                return -1;
            }

            void LRUCache::set(int key, int value) {
                  if(m.find(key)!=m.end()){
                      q.erase(m[key].second);
                      q.push_front(key);
                      m[key] = {value,q.begin()} ;
                  }
                  else{
                      if(q.size()>=cap){
                          int last = q.back();
                          q.pop_back();
                          m.erase(last);
                      }
                       q.push_front(key);
                       m[key] = {value,q.begin()} ;
                  }
            }

            ****************************************************************************************
            ListNode* Solution::solve(ListNode* A, int b) {
                if(A == NULL) return NULL;
                ListNode *head,*curr,*next,*prev;
                curr = A; prev = NULL;
                for(int i = 0; i<b;i++){
                    next = curr->next;
                    curr->next = prev;
                    prev = curr;
                    curr = next;
                }
                head = prev;
                A->next = curr;
                if(curr==NULL) return head;

                for(int i = 0; i<b-1;i++)
                curr = curr->next;

                curr->next = solve(curr->next,b);
                return head;
            }
****************************************************************************************
            ListNode* Solution::reverseBetween(ListNode* A, int B, int C) {
                ListNode* dummy =new ListNode(0);
                dummy->next=A;
                ListNode* prev, *curr, *nextnode;prev=dummy;
                for(int i=0;i<B-1;i++)
                    prev=prev->next;
                curr=prev->next;

                for(int i=0;i<C-B;i++){
                    nextnode=curr->next;
                    curr->next=nextnode->next;
                    nextnode->next=prev->next;
                    prev->next=nextnode;
                }
                return dummy->next;

            }
***************************************************************************************
            ListNode* Solution::reverseList(ListNode* A, int B) {
                if(B==1)
                    return A;
                int c=1;
                ListNode *temp=A, *prev=A, *curr=A->next,*next, *head=NULL, *temp2=NULL;

                while(curr!=NULL){
                    next=curr->next;
                    curr->next=prev;
                    prev=curr;
                    curr=next;
                    c++;
                    if(c%B==0){
                        head= !head?prev:head;
                        if(temp2)
                            temp2->next=prev;
                        temp2=temp;
                        temp=curr;
                    }
                }
                temp2->next=NULL;
                return head;

            }
*****************************************************************************************
            int arr[12345678],minimum[12345678];
            int cur,mincur;
            MinStack::MinStack() {
                cur=-1;
                mincur=-1;
            }

            void MinStack::push(int x) {
                //assert(cur<=1234567);
                if(mincur==-1 || x<=minimum[mincur])
                {
                    mincur++;
                    minimum[mincur]=x;
                }
                cur++;
                arr[cur]=x;
            }

            void MinStack::pop() {
                //assert(cur<=1234567);
                if(cur==-1)
                 return;
                if(arr[cur]==minimum[mincur])
                 mincur--;
                cur--;
            }

            int MinStack::top() {
                //assert(cur<=1234567);
                if(cur==-1)
                 return -1;

                return arr[cur];
            }

            int MinStack::getMin() {
                //assert(cur<=1234567);
                if(mincur==-1)
                 return -1;
                return minimum[mincur];
            }
***************************************************************************************
            int Solution::trap(const vector<int> &A) {
                stack<int>stk;
                for(int i=0;i<A.size();i++){
                    while(!s.empty() and A[i]>A[s.top]){
                        int height = A[s.top()];
                        s.pop();
                        if(s.empty())break;
                        int distance=i-1-s.top();
                        int min_height = min(A[s.top()],A[i])-height;
                        ans+=distance*min_height;

                    }
                    s.push(i);

                }
                return ans;

            }

****************************************************************************************
            string Solution::intToRoman(int A) {
            string numerals[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", 
                         "V", "IV", "I"};
            int values[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};

                if(A<1 && A>3999)
                return "";

                int i=0;
                string res = "";
                while(A > 0)
                {
                    if(A - values[i] >= 0)
                    {
                        res += numerals[i];
                        A = A - values[i];
                    }
                    else
                    {
                        i++;
                    }
                }
                return res;
            }
***************************************************************************************
            string Solution::countAndSay(int A) {
            if(A==1) return "1";
            int i=2;
            string prev="1";
            while(i<=A)
            {
                int count=1;
                char say=prev[0];
                string newp="";
                for(int i=1;i<prev.length();i++)
                    if(prev[i]==prev[i-1]) count++;
                    else{
                        newp+=to_string(count);
                        newp.push_back(say);
                        count=1;
                        say=prev[i];
                    }
                newp+=to_string(count);
                newp.push_back(say);
                i++;
                prev=newp;
            }
            return prev;
            }
*****************************************************************************************
            string Solution::convert(string A, int B)
            {
            vector a(B);
            string p;
            if(B==0) return p;
            int i =0;
            int n = A.size();
            while(i<n)
            {
                int t =0;
                while(t<B and i<n)
                {
                    a[t].push_back(A[i]);
                    i++;
                    t++;
                }
                t = t-2;
                while(t>0 and i<n)
                {
                    a[t].push_back(A[i]);
                    i++;
                    t--;
                }
            }

            string ans;

            for(auto x: a)
                ans.append(x);
            return ans;
            }
***************************************************************************************
            string Solution::multiply(string A, string B) {
            int n = A.length(),m = B.length();
            string res(n+m,'0');

            for(int i=n-1;i>=0;i--){
                for(int j=m-1;j>=0;j--){
                    int num = (A[i] - '0') * (B[j] - '0') + res[i+j+1] - '0';
                    res[i+j+1] = num%10 + '0';
                    res[i+j] += num/10;
                }
            }
            for(int i=0;i<res.length();i++) if(res[i] != '0') return res.substr(i);
            return "0";
            }
*****************************************************************************************
            int Solution::solve(string A) {
            int l=0, r=A.length()-1;
            int count = 0;
            while(l < r){
                if(A[l] == A[r]){
                    l++;
                    r--;
                }else{
                    if(l == 0) {
                        count++;
                        r--;
                    }
                    else {
                        count += l;
                        l = 0;
                    }

                }
            }

            return count;
            }
***************************************************************************************
            vector<int> Solution::solve(TreeNode* A) {
                vector<int>sol;
                queue<TreeNode* >q;
                q.push(A);
                q.push(NULL);
                while(q.size()>1){
                    TreeNode* front=q.front();
                    q.pop();

                    if(!front)
                        continue;
                    if(front->left)
                        q.push(front->left);
                    if(front->right)
                        q.push(front->right);

                    if(q.front()==NULL){
                        sol.push_back(front->val);
                        q.push(NULL);
                    }


                }

                return sol;
            }
****************************************************************************************
            bool getPath(TreeNode *A, int B, vector<int> &path){
                if(!A)return 0;
                path.push_back(A->val);
                if(A->val==B){
                    return 1;
                }
                if(getPath(A->left,B,path) or getPath(A->right,B,path))
                    return 1;
                path.pop_back();

                return 0;
            } 

            vector<int> Solution::solve(TreeNode* A, int B) {
                vector<int>path;
                getPath(A,B,path);
                return path;
            }
***************************************************************************************
            TreeNode* Solution::solve(TreeNode* A) {
                if(A->left == NULL && A->right == NULL)return A;
                if(A->left == NULL)
                    return solve(A->right);
                if(A->right == NULL)
                    return solve(A->left);
                A->left = solve(A->left);
                A->right = solve(A->right);
                return A;

            }
*****************************************************************************************
            void solve(TreeNode *A,int B){
                if(B==0){
                    seti.insert(temp);
                }
                solve(A->left,B-A->val,temp)
            }
            vector<vector<int> > Solution::pathSum(TreeNode* A, int B) {
                solve(A,B);
            }
***************************************************************************************
            vector<vector<int> > Solution::zigzagLevelOrder(TreeNode* A) {
                vector<vector<int> > ans;
                if(!A)return ans;
                queue<TreeNode*>q;
                q.push(A);
                int level=0;
                while(!q.empty()){
                    int size=q.size();
                    vector<int>temp;
                    while(size--){
                        TreeNode *front=q.front();q.pop();
                        temp.push_back(front->val);
                        if(front->left){
                            q.push(front->left);
                        }
                        if(front->right){
                            q.push(front->right);
                        }
                    }
                    if(level%2){
                        reverse(temp.begin(),temp.end());
                    }
                    level++;
                    ans.push_back(temp);
                }
                return ans;
            }
****************************************************************************************
            void solve(TreeNode *root,TreeNode*& head)
            {
                if(!root)
                    return;
                solve(root->right,head);
                solve(root->left,head);
                root->left = NULL;
                root->right = head;
                head = root;
            }
            TreeNode* Solution::flatten(TreeNode* A)
            {
                TreeNode *head = NULL;
                solve(A,head);
                return A;
            }
**************************************************************************************
            void sum0(TreeNode* A,long p, long sum){
                if(!A)return;
                if(!(A->left) and !(A->right)){
                    p=(p*10+A->val)%1003;
                    sum=(sum+p)%1003;
                    return;
                }
                p=(p*10+A->val)%1003;
                sum0(A->left,p,sum);
                sum0(A->right,p,sum);
            }

            int Solution::sumNumbers(TreeNode* A) {
                long sum=0;
                long p=0;
                sum0(A,p,sum);
                return sum;
            }
*****************************************************************************************
            int Solution::solve(vector<int> &pre) {
            stack<int> s;
            int root = INT_MIN;

            for(int i =0;i<pre.size();i++){
                if(root > pre[i]){
                    return 0;
                }
                while(!s.empty() && s.top()<pre[i]){
                   root = s.top();
                   s.pop();
                }
                s.push(pre[i]);
            }
            return 1;
            }
****************************************************************************************
            vector<int> Solution::solve(TreeNode* A, int B) {
                queue<TreeNode*>q;
                q.push(A);
                bool found=0;
                vector<int>ans;
                while(!q.empty()){
                    int s=q.size();
                    bool foundonlevel=0;
                    while(s--){
                        TreeNode* curr=q.front();q.pop();
                        if(found){
                            ans.push_back(v->val);
                            continue;
                        }
                        if((curr->left and curr->left->val==B) or (curr->right and curr->china->val)){
                            foundonlevel=1;
                        }else{
                            if(curr->left)q.push(curr->left);
                            if(curr->right)q.push(curr->right);
                        }
                    }
                    found=foundonlevel;
                }
                return ans;
            }
****************************************************************************************
            int search(int low,int high,vector<int> &post,int val){
                for(int i=low;i<=high;i++){
                    if(post[i]==val)
                        return i;
                }
            }

            TreeNode* solve(vector<int> &in,vector<int> &post,int low,int high,int &index){
                if(low>high)return NULL;
                int mid=search(low,high,post,in[index--]);
                TreeNode *root= new TreeNode(post[mid]);
                root->left=solve(in,post,low,mid-1,index);
                root->right=solve(in,post,mid+1,high,index);

                return root;
            }

            TreeNode* Solution::buildTree(vector<int> &in, vector<int> &post) {
                int n=in.size()-1;
                TreeNode *root;
                int i=n;
                root=solve(in,post,0,n,i);
                return root;
            }
****************************************************************************************
            TreeNode* helper(vector<int> &A,int low,int high){
                if(low>high)return NULL;
                int index=max_element(A.begin()+low,A.begin()+high+1)-A.begin();
                TreeNode* root=new TreeNode(A[index]);
                root->left=helper(A,low,index-1);
                root->right=helper(A,index+1,high);
                return root;
            }

            TreeNode* Solution::buildTree(vector<int> &A) {
                return helper(A,0,A.size()-1);
            }
********************************************************************************************
            void Solution::connect(TreeLinkNode* A) {
                queue<TreeLinkNode*>q;
                q.push(A);
                while(!q.empty()){
                    int c=q.size();
                    TreeLinkNode* prev=q.front();q.pop();
                    if(prev->left) q.push(prev->left);
                    if(prev->right) q.push(prev->right);
                    for(int i=1;i<c;i++){
                        TreeLinkNode *curr=q.front();q.pop();
                        prev->next=curr;
                        prev=curr;
                        if(prev->left)q.push(prev->left);
                        if(prev->right)q.push(prev->right);
                    }
                    prev->next=NULL;
                }
            }
*****************************************************************************************
            bool find(TreeNode *A,int val){
                if(!A)return false;
                if(A->val==val)return true;
                return find(A->left,val)||find(A->right,val);
            }
            TreeNode* LCA(TreeNode *A,int B,int C){
                if(!A)return NULL;
                if(A->val==B or A->val==C)return A;
                TreeNode *L=LCA(A->left,B,C);
                TreeNode *R=LCA(A->right,B,C);
                if(L and R)return A;
                return L?L:R;
            }

            int Solution::lca(TreeNode* A, int B, int C) {
                if(!find(A,B) or !find(A,C))return -1;
                TreeNode *ans=LCA(A,B,C);
                return ans->val;

            }
*****************************************************************************************
            void inorder(TreeNode* A,vector<int> &a){ // inorder traversal of BST gives sorted array
            if(!A)return;
            inorder(A->left,a);
            a.push_back(A->val);
            inorder(A->right,a);
            }
            vector<int> Solution::recoverTree(TreeNode* A) {
            vector<int> a;
            inorder(A,a);
            vector<pair<int,int> > temp;
            for(int i=1;i<a.size();i++)
            if(a[i-1]>a[i])temp.push_back(make_pair(a[i-1],a[i]));
            if(temp.size()==1)return {temp[0].second,temp[0].first}; // this happen when root and its child is swapped
            return {temp[1].second,temp[0].first};
            }
**************************************************************************************
            vector<vector<int> > Solution::verticalOrderTraversal(TreeNode* root) {
                map<int,vector<int>>mp;
                vector<vector<int>>ans;
                if(root==NULL)return ans;
                queue<pair<TreeNode*,int>>q;
                q.push({root,INT_MIN+10000});
                while(!q.empty()){
                    int n=q.size();
                    while(n--){
                        auto cur=q.front();
                        q.pop();
                        mp[cur.second].push_back(cur.first->val);
                        if(cur.first->left)
                            q.push({cur.first->left,cur.second-1});
                        if(cur.first->right)
                            q.push({cur.first->right,cur.second+1});
                    }
                    for(auto x:mp){
                        vector<int>temp;
                        for(auto i:x.second)
                            temp.push_back(i);
                        ans.push_back(temp);
                    }

                }
                return ans;
            }
******************************************************************************************
            int Solution::romanToInt(string A) {
                map<char,int> m;
                m['I'] = 1;
                m['V'] = 5;
                m['X'] = 10;
                m['L'] = 50;
                m['C'] = 100;
                m['D'] = 500;
                m['M'] = 1000;

                int ans = 0, i = 0, N = A.size();
                while(i < N-1){
                    if(m[A[i]]< m[A[i+1]]) ans-=m[A[i]];
                    else ans+=m[A[i]];
                    i++;
                }
                ans+=m[A[i]];
                return ans;
            }
*******************************************************************************************
            int Solution::isNumber(const string A) {
                int n=A.size(), i=0, sign=1, base=0, num_digits=0, num_decimal=0;

                while(A[i]==' ')
                    i++;

                if (A[i] == '-' || A[i] == '+') 
                    i++;

                while((A[i] >= '0' && A[i] <= '9') || A[i]=='.'){
                    if(A[i]=='.')
                        num_decimal++;
                    else
                        num_digits++;
                    i++;
                }

                if(num_digits<1 || num_decimal>1 || A[i-1]=='.')
                    return 0;

                if(A[i]=='e'){
                    i++;
                    if (A[i] == '-' || A[i] == '+')
                        i++;     
                    while(A[i] >= '0' && A[i] <= '9'){
                        num_digits++;
                        i++;
                    }
                    if(num_digits<1)
                        return 0;
                }
                while(A[i]==' ')
                    i++;
                if(i==n)
                    return 1;

                return 0;
*****************************************************************************************
            vector<string> Solution::prettyJSON(string str) {
            vector<string> ans;
            string temp = "";
            int cnt = 0;
            for(int i = 0; i < (int)str.size(); i++) {
                if(str[i] == ' ') continue;
                else if(str[i] == '{') {
                    ans.push_back(temp);
                    ans[(int)ans.size() - 1].push_back('{');
                    cnt++;
                    temp += '\t';
                }
                else if(str[i] == '[') {
                    ans.push_back(temp);
                    ans[(int)ans.size() - 1].push_back('[');
                    cnt++;
                    temp += '\t';
                }
                else if(str[i] == '}') {
                    temp = "";
                    cnt--;
                    for(int j = 0; j < cnt; j++) temp += '\t';
                    ans.push_back(temp);
                    ans[(int)ans.size() - 1].push_back('}');
                }
                else if(str[i] == ']') {
                    temp = "";
                    cnt--;
                    for(int j = 0; j < cnt; j++) temp += '\t';
                    ans.push_back(temp);
                    ans[(int)ans.size() - 1].push_back(']');
                }
                else if(str[i] == ',') {
                    ans[(int)ans.size() - 1].push_back(str[i]);
                }
                else {
                    ans.push_back(temp);
                    while(str[i] != '{' && str[i] != '}' && str[i] != '[' && str[i] != ']' && str[i] != ',') {
                        ans[(int)ans.size() - 1].push_back(str[i]);
                        i++;
                    }
                    i--;
                }
            }
            return ans;
            }
*****************************************************************************************
            vector Solution::fullJustify(vector &A, int B) {
            vector <string> ans;

            for(int i=0;i<A.size();){
                int s = 0,n = 1,old_i = i,j;
                string line = "";
                s = A[i].size();
                i++;
                /*  n = no of words in a line
                    n-1  = no of gaps 
                    s = max possible lenght with 1 spacing in a line 
                */
                while(((s+A[i].size()+n)<=B)&&i<A.size()){
                    s = s + A[i].size();
                    i++;
                    n++;
                }
                //single element in a line 
                if(n==1){
                    line  = A[i-1];
                    line = line +string(B-line.size(),'  ');
                    ans.push_back(line);
                    continue;
                }
                //last line 
                if(i == A.size()){
                    for(j = old_i;j<i-1;j++) 
                        line = line + A[j] + '  ';
                    line = line + A[j];
                    line = line + string(B-line.size(),'  ');
                    ans.push_back(line);
                    break;
                }
                //ususal case
                int q = (B-s)/(n-1);
                int r = (B-s)%(n-1) ;
                for(j = old_i;j<i-1;j++){
                    line = line + A[j] ;
                    if(r) {
                        line = line + string(q+1,'  ');
                        r--;
                    }
                    else
                        line = line + string(q,'  ');
                }
                line = line + A[j];
                ans.push_back(line);
            }
            return ans;
            }
***************************************************************************************
            vector Solution::solve(TreeNode* A) {
            deque<TreeNode* >s;
            vectorv;
            s.push_front(A);
            TreeNode* root;
            while(!s.empty()){
            root=s.front();
            s.pop_front();
            v.push_back(root->val);
            if(root->left)
            s.push_back(root->left); //we have to traverse left node at end.
            if(root->right)
            s.push_front(root->right); // we have to traverse right node first.

            }
            return v;
            }
****************************************************************************************
            vector vec;
            void allv(TreeNode* node,vector &vec){
            if(node==NULL){
            return;
            }
            allv(node->left,vec);
            vec.push_back(node->val);
            allv(node->right,vec);
            }
            BSTIterator::BSTIterator(TreeNode *root) {

              allv(root,vec);
            }

            /** @return whether we have a next smallest number */
            bool BSTIterator::hasNext() {
            if(vec.size()!=0){
            return true;
            }
            else{
            return false;
            }
            }

            /** @return the next smallest number */
            int BSTIterator::next() {
            int c=vec[0];
            vec.erase(vec.begin()+0);
            return c;

            }
********************************************************************************************
            TreeNode * build(vector<int> &pre, int &preIdx,unordered_map<int,int> &m,int start,int end){
                if(start>end) return NULL;
                TreeNode *root = new TreeNode(pre[preIdx++]);
                if(start == end) return root;
                root->left = build(pre,preIdx,m,start,m[root->val]-1);
                root->right = build(pre,preIdx,m,m[root->val]+1,end);
                return root;
             }
            TreeNode* Solution::buildTree(vector<int> &pre, vector<int> &in) {
                unordered_map<int,int> mp;
                for(int i=0; i<in.size(); i++) mp[in[i]] = i;
                int n=0;
                return build(pre,n,mp,0,in.size()-1);
            }
***************************************************************************
            class Trie{
            private:
            class Node{
            public:
            bool end;
            Node* child[26];
            int count;
            Node()
            {
            end=false;
            count=0;
            memset(child, NULL, sizeof(child));
            }

            };

            public:
            Node* root;
            Trie()
            {
                root=new Node();
            }
            void insert(string word)
            {
                Node* temp=root;
                for(auto c:word)
                {
                    if(!temp->child[c-'a'])
                    temp->child[c-'a']=new Node();
                    temp=temp->child[c-'a'];
                    temp->count++;
                }
                temp->end=true;
            }
            string search(string word)
            {
                Node* temp=root;
                string ans;
                for(auto c: word)
                { 
                    ans.push_back(c);
                    temp=temp->child[c-'a'];
                    if(temp->count==1)
                    break;
                }
                return ans;
            }
            };

            vector Solution::prefix(vector &A) {
            int n=A.size();
            Trie trie;
            for(auto word: A)
            {
            trie.insert(word);
            }
            vectorres;
            for(auto word: A)
            {
            res.push_back(trie.search(word));
            }
            return res;
            }                    
***************************************************************************
            bool compare(const pair<int, int> &a, const pair<int, int> &b) {
                if (a.first == b.first) {
                    return (a.second < b.second);
                } else {
                    return (a.first > b.first);
                }
            }

            vector<int> Solution::solve(string A, vector<string> &B) {
                unordered_map<string,int> mp;

                for(int i=0;i<A.size();i++)
                {
                    string s;
                    while(A[i]!='_' && i<A.size())
                    {
                        s.push_back(A[i]);
                        i++;
                    }
                    mp[s]++;
                }
                vector<pair<int,int> > vp;
                for(int i=0;i<B.size();i++)
                {
                    int count=0;
                    for(int j=0;j<B[i].size();j++)
                    {
                        string s;
                        while(B[i][j]!='_' && j<B[i].size())
                        {
                            s.push_back(B[i][j]);
                            j++;
                        }
                        if(mp.find(s)!=mp.end()) count++;
                    }
                    vp.push_back(make_pair(count,i));
                }
                sort(vp.begin(),vp.end(), compare );
                vector<int> ans;
                for(int i=0;i<vp.size();i++)
                {
                    ans.push_back(vp[i].second);
                }
                return ans;

            }
***************************************************************************
            vector Solution::order(vector &A, vector &B) {
            int k=0,j;
            vector<pair<int,int>> vect;
            for(int i=0;i<A.size();i++)
            {
            vect.push_back(make_pair(A[i],B[i]));
            }
            sort(vect.begin(),vect.end());
            reverse(vect.begin(),vect.end());
            pair<int,int> temp;
            for(int i=1;i<A.size();i++)
            {
            j=i;
            if(vect[i].second!=i)
            {
            temp=vect[j];
            while(temp.second!=j)
            {
            vect[j]=vect[j-1];
            j–;
            }
            vect[j]=temp;
            }
            }
            for(int i=0;i<A.size();i++)
            {
            A[i]=vect[i].first;
            }
            return A;
            }

**************************************************************************
            vector<long long int>v;long long int m;
                int f(long long int x){
                    int cnt=1,p=0;
                    for(int i=0;i<v.size();i++){
                        if(v[i]>x)return 0;
                        if(v[i]+p<=x)p+=v[i];
                        else p=v[i],cnt++;
                    }
                    return cnt<=m;
                }
                int Solution::books(vector<int> &A, int B) {
                    if(B>A.size())return -1;
                    v=vector<long long int>(A.size());m=B;
                    for(int i=0;i<A.size();i++)v[i]=A[i];
                    long long int l=0,r=1e18;
                    while(r-l>1){
                        long long int m=l+r>>1;
                        if(f(m))r=m;
                        else l=m;
                    }
                    if(f(l+1))return (int)l+1;
                    if(f(l))return (int)l;
                    return -1;
                }
**************************************************************************
            void put_max(vector<pair<int,int>> & v,int l,int r){
                long long mod = 1e9+7;
                stack<pair<int,int>> st;
                for(int i=l;i<=r;i++){
                    while(st.size() && (v[i].first >= st.top().first)){
                        st.pop(); 
                    }
                    v[i].second = st.empty() ? (i+1) : (i - st.top().second);
                    st.push({v[i].first,i});
                }

                while(st.size())
                    st.pop();

                for(int i=r;i>=l;i--){
                    while(st.size() && v[i].first > st.top().first){
                        st.pop();
                    }
                    int gt = r - i + 1;
                    if(st.size()){
                        gt = st.top().second - i;
                    }
                    v[i].second = (v[i].second * gt) % mod;
                    st.push({v[i].first,i});
                }
            }
            vector<int> Solution::solve(vector<int> &A, vector<int> &B) {
                vector<pair<int,int>> v(A.size()),q;
                for(int i=0;i<A.size();i++){
                    v[i] = {A[i],0};
                }
                put_max(v,0,A.size()-1);
                // for(auto & i:v){
                //     // cout << i.first << "," << i.second << " ";
                // }
                // cout << endl;
                long long mod = 1e9+7;
                int id = 1;
                for(auto & pr:v){
                    int ele = pr.first;
                    long long num=1;
                    int i;
                    for(i=1;i*i<ele;i++){
                        if(ele%i==0){
                            num = (num * i)%mod;
                            num = (num * (ele/i))%mod;
                        }
                    }
                    if((i*i)==ele)
                        num = (num * i)%mod;
                    pr.first = num;
                }
                sort(v.rbegin(),v.rend());
                for(int i=0;i<B.size();i++){
                    q.push_back({B[i],i});
                }
                sort(q.begin(),q.end());
                long long ind = 0,mind=v[0].second;
                for(auto & i:q){
                    // cout << i.first << "," << i.second << " ";
                    while(i.first > mind){
                        mind+=v[++ind].second;
                    }
                    i.first = v[ind].first;
                    swap(i.first,i.second);
                }
                // cout<< endl;
                sort(q.begin(),q.end());
                for(int i=0;i<B.size();i++){
                    B[i] = q[i].second;
                }
                return B;
            }
***************************************************************************
            bool isPossible(int A, int B, vector<int> &C,long long int X){
                int n=C.size();
               long long int t=X;
                int i=0,cnt=1;
                while(i<n){
                    if(cnt>A)
                     return false;
                    if(C[i]>t){
                        cnt++;
                        t=X;
                    }
                    else{
                        t=t-C[i];
                        i++;
                    }
                }
                return true;
            }

            int Solution::paint(int A, int B, vector<int> &C) {
                int n=C.size();
                long long int sum=0;
                for(int i=0;i<n;i++)
                 sum=sum%10000003+C[i]%10000003;
                long long int low=0,high=sum*B;
                long long int ans=high%10000003;
                while(low<=high){
                    //cout<<low<<" "<<high<<" "<<ans<<endl;
                    long long int mid=low+(high-low)/2;
                    if(isPossible(A,B,C,mid/B)){
                       // cout<<"Hi\n";
                        ans=mid%10000003;
                        high=mid-1;
                    }
                    else low=mid+1;
                }
                return ans%10000003;
            }
***************************************************************************
            void print(vector<bool> &vec) {
                for(auto it: vec)   cout<<it<<" ";  cout<<endl;
            }
            int vec2int(vector<bool> &vec) {
                int val = 0;
                for(auto it: vec)
                    val = (val<<1) + it;
                return val;
            }
            void genPalin(vector<bool> &vec, set<int> &s, int pos) {
                int len = vec.size();
                if((len%2 == 0 and pos >= len/2) or (len%2 and pos > len/2)) {
                    // print(vec);
                    s.insert(vec2int(vec));
                    return;
                }
                genPalin(vec, s, pos+1);

                vec[pos] = vec[len-1-pos] = false;
                genPalin(vec, s, pos+1);
                vec[pos] = vec[len-1-pos] = true;
            }
            int Solution::solve(int A) {
                set<int> s;
                for(int i=1; i<32; i++) {
                    vector<bool> vec(i, true);
                    genPalin(vec, s, 1);
                    if(s.size() >= A)   break;
                }
                auto it = s.begin();
                int val = 1;
                for(int i=1; i<=A; i++)
                    val = *it++;
                return val;
            }
***************************************************************************
            int Solution::divide(int A, int B) {
                int ans=0;
                while(B>=1){
                    if(B&1==1) ans+=A;
                    B=B>>1;
                    A+=A;
                }
                return ans;
            }
**************************************************************************
            int Solution::isScramble(const string A, const string B) {
            if(A.length()!=B.length())
            return 0;
            int n=A.length();
            if(n==0)
            return 1;
            if(A==B)
            return 1;
            string copy_A=A,copy_B=B;
            sort(copy_A.begin(),copy_A.end());
            sort(copy_B.begin(),copy_B.end());
            if(copy_A!=copy_B)
            return 0;
            for(int i=1;i<n;i++)
            {
            if(isScramble(A.substr(0,i),B.substr(0,i))&&isScramble(A.substr(i,n-i),B.substr(i,n-i)))
            return 1;
            if(isScramble(A.substr(0,i),B.substr(n-i,i))&&isScramble(A.substr(i,n-i),B.substr(0,n-i)))
            return 1;
            }
            return 0;

            }
**************************************************************************
            unordered_map<string,int>mp;

            int solve(string s,int i,int j, bool isTrue){
            if(i>j)
            return false;
            if(i==j){
            if(isTrue==true)
            return s[i]==‘T’;
            else
            return s[i]==‘F’;
            }

            string temp=to_string(i)+" "+to_string(j)+" "+to_string(isTrue);

            if(mp.find(temp)!=mp.end())
                return mp[temp];

            int ans=0;

            for(int k=i+1;k<=j-1;k+=2){

                int lt=solve(s,i,k-1,true);
                int lf=solve(s,i,k-1,false);
                int rt=solve(s,k+1,j,true);
                int rf=solve(s,k+1,j,false);

                if(s[k]=='&'){
                    if(isTrue==true)
                        ans=ans + lt*rt;
                    else
                        ans=ans + lf*rt + lt*rf + lf*rf;
                }

                if(s[k]=='|'){
                    if(isTrue==true)
                        ans= ans + lt*rt + lt*rf + lf*rt;
                    else
                        ans=ans+ lf*rf;
                }

                if(s[k]=='^'){
                    if(isTrue==true)
                        ans=ans+ lf*rt + lt*rf;
                    else
                        ans=ans+ lt*rt + lf*rf;
                }
            }

            ans=ans%1003;
            return mp[temp]=ans;
            }

            int Solution::cnttrue(string A) {
            mp.clear();
            return solve(A,0,A.length()-1,true);
            }
***************************************************************************
            int overlap(string a,string b,string& res){
                string x=a+"#"+b;
                vector<int> kmp(x.length());
                for(int i=1,j=0;i<x.length();i++){
                    if(x[i]==x[j]){
                        j++;
                        kmp[i]=j;
                    }
                    else {
                        while(j){
                            j=kmp[j-1];
                            if(x[i]==x[j])
                            break;
                        }
                        if(x[i]==x[j]){
                            j++;
                            kmp[i]=j;
                        }
                        else kmp[i]=j;
                    }
                }
                int ans=kmp[x.length()-1];
                res=b.substr(0,b.length()-ans)+a;
                return ans;
            }
            int Solution::solve(vector<string> &A) {
                while(A.size()!=1){
                    int l,r,lap=INT_MIN,temp;
                    string str,rep;
                    for(int i=0;i<A.size();i++){
                        for(int j=i+1;j<A.size();j++){
                           temp=overlap(A[i],A[j],rep);
                           if(temp>lap){
                               lap=temp;str=rep;l=i;r=j;
                           }
                           temp=overlap(A[j],A[i],rep);
                           if(temp>lap){
                               lap=temp;str=rep;l=i;r=j;
                           }               
                        }
                    }
                    A.erase(A.begin()+l);A.erase(A.begin()+r-1);
                    A.push_back(str);
                }
                return A[0].length();
            }
***************************************************************************
            int dp[1000][1000];
            int dfs(int ind, string s, int k)
            {
              if(ind==s.size() && k==0)
                return 0;
              if(k<=0)
                return -1;
              if(dp[ind][k]!=-1)
                return dp[ind][k];
              int w=0,b=0,mini=INT_MAX;
              for(int i=ind;i<s.size();i++)
              {
                if(s[i]=='W')
                  w++;
                else
                  b++;
                int temp=dfs(i+1,s,k-1);
                if(temp!=-1 && temp!=INT_MAX)
                  mini=min(mini,temp+w*b);
              }
              dp[ind][k]=mini;
              return mini;
            }
            int Solution::arrange(string A, int B)
            {
              if(B>A.size())
                return -1;
              memset(dp,-1,sizeof(dp));
              return dfs(0,A,B);

            }

**************************************************************************
            int subwithzero(vector<int> &A){
            int sum=0;
            int count = 0;
            unordered_map<int,int> mp;
            mp[0] = 1;
            for(int i=0;i<A.size();i++){
                sum+=A[i];
                if(mp.find(sum)!=mp.end()) count+=mp[sum];
                mp[sum]++;
            }
            return count;
            }

            int Solution::solve(vector<vector<int> > &A) {
            int count = 0;
            int n = A.size();
            int m = A[0].size();
            if(n==0 && m==0) return 0;
            for(int i=0;i<m;i++){
                vector<int> temp(n,0);
                for(int j=i;j<m;j++){

                    for(int k=0;k<n;k++){
                        temp[k] += A[k][j];
                    }
                    count+=subwithzero(temp);
                }
            }
            return count;
            }
**************************************************************************
            #define MOD 1000000007ll
            typedef long long LL;

            //adds y to x, modulo MOD
            void add(int &x, int y){
                x += y;
                if(x>=MOD)x-=MOD;
            }

            //choose and dp tables
            vector< vector<int > > choose,dp;

            //build choose table
            void build(int N){
                choose[0][0]=1;
                for(int i=1; i<=2*N; i++){
                    choose[i][0]=1;
                    for(int j=1; j<=i; j++){
                        choose[i][j]=choose[i-1][j];
                        add(choose[i][j], choose[i-1][j-1]);
                    }
                }
            }

            //reccurence function as defined in hint_2
            int rec(int n, int h){ 
                if(n<=1)return (h==0);
                if(h<0)return 0;
                int &ret=dp[n][h];
                if(ret!=-1)return ret;
                ret=0;
                int x, y;
                for(int i=1; i<=n; i++){
                    x=i-1;
                    y=n-x-1;
                    int sum1=0,sum2=0,ret1=0;
                    for(int j=0; j<=h-2; j++){
                        add(sum1, rec(x, j));
                        add(sum2, rec(y, j));
                    }
                    add(ret1, ((LL)sum1*(LL)rec(y, h-1))%MOD);
                    add(ret1, ((LL)sum2*(LL)rec(x, h-1))%MOD);
                    add(ret1, ((LL)rec(x, h-1)*(LL)rec(y, h-1))%MOD);
                    ret1 = ((LL)ret1*(LL)choose[x+y][y])%MOD;
                    add(ret, ret1);
                }
                return ret;
            }

            int Solution::cntPermBST(int A, int B) {
                int n=50;
                choose.clear();
                dp.clear();
                choose.resize(2*n+1,vector<int>(2*n+1, 0));
                dp.resize(n+1,vector<int>(n+1, -1));
                build(n);
                return rec(A, B);
            }
***************************************************************************
            int hist_area(vector<int> v) {
                sort(v.begin(), v.end());
                int area = 0;    

                int n = v.size();

                int i = 0;

                for(i = 0; i < n; i++) {
                    area = max(area, v[i] * (n - i));
                }

                return area;
            }
            int Solution::solve(vector<vector<int> > &A) {
                int n = A.size();
                if(n == 0) return 0;

                int m = A[0].size();

                vector<int> v = A[0];

                int area = hist_area(v);

                for(int i = 1; i < n; i++) {
                    for(int j = 0; j < m; j++) {
                        if(A[i][j] == 0) v[j] = 0;
                        else v[j]++;
                    }   
                    area = max(area, hist_area(v));
                }
                return area;
            }

**************************************************************************
            int Solution::solve(const vector<int> &A){
            int n = A.size();
            int sumv = 0;
            for(int i=0;i<n;i++){
                sumv += A[i];
            }
            int req = sumv/2;
            vector<vector<int> > dp(n+1,vector<int>(req+1,INT_MAX));
            dp[0][0] = 0;
            for(int i=0;i<=n;i++){
                dp[i][0] = 0;
            }
            for(int i=1;i<=n;i++){
                for(int j=1;j<=req;j++){
                    int a=INT_MAX;
                    int b;
                    if(j>=A[i-1]){
                        a = dp[i-1][j-A[i-1]];
                        if(a!=INT_MAX)a+=1;
                    }
                    b = dp[i-1][j];
                    dp[i][j] = min(a,b);
                }
            }
            for(int i=req;i>=0;i--){
                if(dp[n][i]!=INT_MAX){
                    return dp[n][i];
                }
            }
            }
**************************************************************************
            vector Solution::solve(int S, vector &wt) {
            int n=wt.size();
            vector<int> dp(S+1,-1),back(S+1);

            for(int i=0;i<=S;i++){ // in normal unbounded we iterate over every Sum by including item one by one

                for(int j=0;j<n;j++){//this order is changed for lexographically minimum

                    if(i>=wt[j] && dp[i]<dp[i-wt[j]]+1){
                        dp[i]=dp[i-wt[j]]+1;
                        back[i]=j;
                    }

                }
            }

            vector<int> r;
            while(S>=0 && S-wt[back[S]]>=0){
                r.push_back(back[S]);
                S-= wt[back[S]];
            }
            return r;
            }
**************************************************************************
            int findMaxRectangle(vector<int> A){
                int n = A.size();
                stack<int> s;
                int l[n]; int r[n];

                for(int i=0;i<n;i++){
                    while(!s.empty() && A[s.top()] >= A[i]){
                        s.pop();
                    }
                    if(s.empty()) l[i] = -1;
                    else l[i] = s.top();
                    s.push(i);
                }
                while(!s.empty()){
                    s.pop();
                }

                for(int i=n-1;i>=0;i--){
                    while(!s.empty() && A[s.top()] >= A[i]){
                        s.pop();
                    }
                    if(s.empty()) r[i] = n;
                    else r[i] = s.top();
                    s.push(i);
                }
                int ret = 0;
                for(int i=0;i<n;i++){
                    int cur = A[i]*(r[i]-l[i]-1);
                    ret = max(cur,ret);
                }
                return ret;
            }

            int Solution::maximalRectangle(vector<vector<int> > &A) {
                int m = A.size();
                int n = A[0].size(),maxArea = 0;
                for(int i=1;i<m;i++)
                for(int j=0;j<n;j++){
                    if(A[i][j]==1) A[i][j] += A[i-1][j];
                }
                for(int i=0;i<m;i++){
                    int curArea = findMaxRectangle(A[i]);
                    maxArea = max(curArea,maxArea);
                }
                return maxArea;
            }
***************************************************************************
            vector<vector> average(vector a)
            {
            int total = 0;
            for(int i = 0; i < a.size(); i++)
            total += a[i];

            vector<vector<int>> arr(total+1);
            for(int i=0;i<=total;i++) arr[i].resize(a.size()+1,INT_MAX);
            for(int i=0;i<=total;i++)
            {
                for(int j=0;j<=a.size();j++)
                {
                    if((j!=0 && j!=a.size()) && (i*(a.size()-j) ==(total-i)*j)) arr[i][j] = j;
                }
            }
            for(int k=a.size()-1;k>=0;k--)
            {
                for(int i=0;i<=total;i++)
                {
                    for(int j=0;j<=a.size();j++)
                    {
                        if((j!=0 && j!=a.size()) && (i*(a.size()-j) ==(total-i)*j)) arr[i][j] = j;
                        if(j==a.size() || i+a[k]>total) continue;
                        else arr[i][j] = min(arr[i+a[k]][j+1],arr[i][j]);
                    }
                }
            }
            vector<vector<int>> ans;
            if(arr[0][0]==INT_MAX) return ans;
            int i = 0;
            int j = 0;
            int k = 0;
            ans.resize(2);
            while(1)
            {
                if(k==a.size()) return ans;
                int op1 = arr[i+a[k]][j+1];
                int op2 = arr[i][j];
                if(op1<=op2) {ans[0].push_back(a[k]); i+=a[k]; j++; k++;}
                else {ans[1].push_back(a[k]); k++;}
            }
            return ans;
            }

            vector<vector > Solution::avgset(vector &A) {
            sort(A.begin(), A.end());
            return average(A);
            }
***************************************************************************
            vector<vector<string> > ans;
            int min_dist;
            string dest;
            bool isAdjacent(string A,string B)
            {
                int count=0;
                for(int i=0;i<A.length();i++)
                {
                    if(A[i]!=B[i])
                        count++;
                    if(count>1)
                        return false;
                }
                return ((count==1)?true:false);
            }
            void dfs(vector<string> path,string vertex,int dist,unordered_set<string> dict)
            {
                if(dist>min_dist)
                    return;
                if(vertex==dest)
                {
                    if(dist==min_dist)
                        ans.push_back(path);
                    return;
                }
                dict.erase(vertex);
                for(auto it=dict.begin();it!=dict.end();++it)
                    if(isAdjacent(vertex,*it))
                    {
                        path.push_back(*it);
                        dfs(path,*it,dist+1,dict);
                        path.pop_back();
                    }
                dict.insert(vertex);
            }
            vector<vector<string> > Solution::findLadders(string start, string end, vector<string> &dict) {
                // Do not write main() function.
                // Do not read input, instead use the arguments to the function.
                // Do not print the output, instead return values as specified
                // Still have a doubt. Checkout www.interviewbit.com/pages/sample_codes/ for more details
                min_dist=INT_MAX;dest=end;ans.clear();
                unordered_set<string> dictMap(dict.begin(),dict.end());
                dictMap.insert(end);
                queue<pair<string,int> > bfs_queue;
                unordered_map<string,bool> visited;
                visited[start]=true;
                bfs_queue.push(make_pair(start,0));
                while(!bfs_queue.empty())
                {
                    pair<string,int> vertex=bfs_queue.front();
                    bfs_queue.pop();
                    if(vertex.first==end)
                    {
                        min_dist=vertex.second;
                        break;
                    }
                    dictMap.erase(vertex.first);
                    for(auto it=dictMap.begin();it!=dictMap.end();++it)
                        if(isAdjacent(vertex.first,*it))
                            bfs_queue.push(make_pair(*it,vertex.second+1));
                }
                if(min_dist==INT_MAX)
                    return ans;
                vector<string> path;
                path.push_back(start);
                dictMap.clear();
                dictMap.insert(dict.begin(),dict.end());
                dictMap.insert(end);
                dfs(path,start,0,dictMap);
                return ans;
            }
***************************************************************************
            int Solution::solve(string A, string B, vector<string> &C) 
            {
                queue<string>q;
                q.push(A);
                int depth = 0;
                int levelSize = 0;
                set<string>s;
                for(int i=0;i<C.size();i++)
                    s.insert(C[i]);

                while(!q.empty())
                {
                    depth++;
                    levelSize = q.size();
                    while(levelSize--)
                    {
                        string word = q.front();
                        q.pop();

                        for(int i=0;i<word.length();i++)
                        {
                            string temp = word;
                            for(char c='a';c<='z';c++)
                            {
                                temp[i] = c;
                                if(temp.compare(word) == 0)//skip the same word
                                    continue;
                                if(temp.compare(B) == 0)
                                    return depth + 1;
                                if(s.find(temp) != s.end())
                                {
                                    q.push(temp);
                                    s.erase(temp);
                                }
                            }
                        }
                    }
                }

                return 0;
            }
**************************************************************************
            bool cmp(vector &a,vector &b){
            return a[2] < b[2];
            }
            int find(int p[],int a){
            if(p[a] == -1) return a;
            return p[a] = find(p,p[a]);
            }
            int Solution::solve(int A, vector<vector > &B) {
            int par[A+10], ans=0;
            memset(par, -1, sizeof(par));
            sort(B.begin(), B.end(),cmp);
            for(auto i : B) {
            int y = find(par, i[0]), x = find(par, i[1]);
            if(x != y){
            par[x] = y;
            ans += i[2];
            if(A-- == 1) return ans;
            }
            }
            return ans;
            }
**************************************************************************
            RandomListNode* deepCopy( \
                                        RandomListNode* A, \
                                        unordered_map<RandomListNode *, RandomListNode *> &umap) {
                if (not A)   return nullptr;
                if (umap.find(A) == umap.end()) {
                    RandomListNode *temp = new RandomListNode (A->label);
                    umap.insert({A, temp});
                    temp->next = deepCopy (A->next, umap);
                    temp->random = deepCopy (A->random, umap);
                }
                return umap[A];
            }
            RandomListNode* Solution::copyRandomList(RandomListNode* A) {
                unordered_map<RandomListNode *, RandomListNode *> umap;
                return deepCopy(A, umap);
            }
**************************************************************************                              
            typedef long long ll;
            string Solution::fractionToDecimal(int A, int B) {
                if(A==0) return "0";
                if(B==0) return "invalid";
                long long a=(long long)A;
                long long b=(long long)B;
                int sign=(a<0)^(b<0)?-1:1;
                a=abs(a);
                b=abs(b);
                long long initial=a/b;
                string res;
                if(sign==-1)
                {
                res+="-";
                }
                res+=to_string(initial);
                if(a%b==0)
                {
                    return res;
                }
                res+=".";
                long long rem=a%b;
                unordered_map<ll,ll> m;
                long long index;
                bool repeat =false;
                while(rem>0&&repeat==false)
                {
                    if(m.find(rem)!=m.end())
                    {
                        index=m[rem];
                        repeat=true;
                        break;
                    }
                    else
                    {
                        m[rem]=res.size();
                    }
                    rem=rem*10;
                    long long temp=rem/b;
                    res+=to_string(temp);
                    rem=rem%b;
                }
                    if(repeat==true)
                    {
                        res+=")";
                        res.insert(index,"(");
                    }
                    return res;
            }
**************************************************************************  
            long long dp[105];
            long long comb[105][105];
            #define MOD 1000000007
            int rec(int n)
            {
                if(n<=1)
                    return 1;
                if(dp[n] != -1)
                    return dp[n];
                int i;
                int fill = 0;
                int pw = 2;
                int left,right;
                left = right = 0;

                while(fill+pw < n-1)
                {
                    fill += pw;
                    left += pw/2;
                    right += pw/2;
                    pw *= 2;
                }
                int rem = n-1-fill;
                if(rem > pw/2)
                {
                    left += pw/2;
                    right += (rem-pw/2);
                }
                else
                    left += rem;

                return dp[n] = (((rec(left)*1LL*rec(right))%MOD)*1LL*comb[n-1][left])%MOD;
            }
            int Solution::solve(int A)
            {
                int i,j;
                for(i=0;i<=100;i++)
                    dp[i] = -1;
                comb[0][0] = 1;
                for(i=1;i<=100;i++)
                {
                    comb[i][0] = comb[i][i] = 1;
                    for(j=1;j<i;j++)
                        comb[i][j] = (comb[i-1][j-1]+comb[i-1][j])%MOD;
                }
                return rec(A);
            }
**************************************************************************  
            ListNode* Solution::insertionSortList(ListNode* A) {

            priority_queue <int, vector<int>, greater<int>> pq;
            ListNode*itr = A;
            pq.push(A->val);
            while(itr->next!=NULL)
            {

                itr = itr->next;
                pq.push(itr->val);
            }
            itr=A;
            itr->val = pq.top();
            pq.pop();
            while(itr->next)
            {

                itr = itr->next;
                itr->val = pq.top();
                pq.pop();
            }

            return A;
            }
***************************************************************************  
            ListNode* mergeTwoLists(ListNode* A, ListNode* B) {
            if(A==NULL)return B;
            if(B==NULL)return A;
            ListNode* ans=NULL;
            if(A->val < B->val){
            ans=A;
            ans->next=mergeTwoLists(A->next,B);
            }
            else{
            ans=B;
            ans->next=mergeTwoLists(A,B->next);
            }
            return ans;
            }
            ListNode* Solution::sortList(ListNode* A) {
            if(A==NULL||A->next==NULL)return A;
            ListNode* slow=A;
            ListNode* temp=A;
            ListNode* fast=A;
            while(fast!=NULL && fast->next!=NULL){
            temp = slow;
            slow = slow->next;
            fast=fast->next->next;
            }

            temp->next = NULL;
             ListNode* LS = sortList(A);
             ListNode* LR = sortList(slow);

            return mergeTwoLists(LS,LR);
            }
***************************************************************************  
            int Solution::largestRectangleArea(vector<int> &A) {

               set<int> s;
               for(auto x:A)s.insert(x);
                int area=0;
                for(auto x:s){
                    int count=0;
                    for(auto i:A){
                        if(i<x)count=0;
                        else {count++;
                          area=max(x*count,area);
                        }
                    }
                }
                return area;
            }
