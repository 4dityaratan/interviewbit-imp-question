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
*****************************************************************************************
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

*****************************************************************************************
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
*****************************************************************************************
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

******************************************************************************************
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

****************************************************************************************
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

*****************************************************************************************
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

*****************************************************************************************
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

*****************************************************************************************
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

*****************************************************************************************
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
******************************************************************************************
****************************************************************************************
*****************************************************************************************
*****************************************************************************************
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

*****************************************************************************************
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
*****************************************************************************************
******************************************************************************************
****************************************************************************************
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
*****************************************************************************************
*****************************************************************************************
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

*****************************************************************************************
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

*****************************************************************************************
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

******************************************************************************************
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

****************************************************************************************
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
*****************************************************************************************
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

******************************************************************************************
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

****************************************************************************************
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

*****************************************************************************************
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

******************************************************************************************
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

****************************************************************************************
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

*****************************************************************************************
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

******************************************************************************************
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
****************************************************************************************
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

*****************************************************************************************
int Solution::solve(vector<int> &A) {
    vector<int>hgt(A.size(),0);
    int ans=0,maxx=0;
    for(int i=A.size()-1;i>0;i--){
        ans=max(ans,hgt[A[i]]+hgt[i]+1);
        hgt[A[i]]=max(hgt[i]+1,hgt[A[i]]);
    }
    return ans;
}

******************************************************************************************
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

****************************************************************************************
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
    
*****************************************************************************************
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

******************************************************************************************
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
******************************************************************************************
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
****************************************************************************************
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
******************************************************************************************
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
****************************************************************************************
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

******************************************************************************************
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
****************************************************************************************
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
******************************************************************************************
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
****************************************************************************************
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
******************************************************************************************
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
****************************************************************************************
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
******************************************************************************************
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
****************************************************************************************
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
******************************************************************************************
void solve(TreeNode *A,int B){
    if(B==0){
        seti.insert(temp);
    }
    solve(A->left,B-A->val,temp)
}
vector<vector<int> > Solution::pathSum(TreeNode* A, int B) {
    solve(A,B);
}
****************************************************************************************
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
            ******************************************************************************************
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
            ****************************************************************************************
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
            ******************************************************************************************
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
            ******************************************************************************************
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
            *****************************************************************************************
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
            ******************************************************************************************
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
            *********************************************************************************************
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
            ******************************************************************************************
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
            ******************************************************************************************
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
            *****************************************************************************************
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
*********************************************************************************************
******************************************************************************************
******************************************************************************************
*****************************************************************************************
******************************************************************************************
*********************************************************************************************
****************************************************************************
